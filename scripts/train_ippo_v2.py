#!/usr/bin/env python3
from __future__ import annotations
"""Eksperimen V2: Square Formation + Smoothness Penalty.
Cara pakai: terminal kedua > python3 train_ippo_v2.py
Output: models/smoothness/ppo_square_final.zip"""

import os
import time
import numpy as np
import rospy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_gazebo import GazeboFormationEnv, ROBOT_NAMES

# KONFIGURASI TRAINING
TOTAL_TIMESTEPS  = 100_000
SAVE_EVERY_STEPS = 50_000 # checkpoint
LOG_INTERVAL     = 10

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/smooth_v2")
LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/smoothness")

# PPO Hyperparameters
PPO_PARAMS = {
    "learning_rate" : 3e-4,
    "n_steps"       : 1024, # per env — total buffer = 1024 × 4 robot = 4096
    "batch_size"    : 64,
    "n_epochs"      : 10,
    "gamma"         : 0.99,
    "gae_lambda"    : 0.95,
    "clip_range"    : 0.2,
    "ent_coef"      : 0.01,
    "vf_coef"       : 0.5,
    "max_grad_norm" : 0.5,
    "verbose"       : 1,
}

POLICY_KWARGS = {
    "net_arch" : [64, 64],
}


#  CALLBACK
class SharedPPOCallback(BaseCallback):
    """Checkpoint + logging untuk shared policy."""

    def __init__(self, save_dir: str, save_every: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir    = save_dir
        self.save_every  = save_every
        self._last_save  = 0
        self._ep_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps - self._last_save >= self.save_every:
            path = os.path.join(
                self.save_dir,
                f"ppo_square_smooth_step{self.num_timesteps}"
            )
            self.model.save(path)
            self._last_save = self.num_timesteps

            if self._ep_rewards:
                mean_r = np.mean(self._ep_rewards[-50:])
                print(f"  [shared] step={self.num_timesteps:,} | "
                      f"mean_reward(last50ep)={mean_r:.3f} | "
                      f"saved → {path}.zip")
            else:
                print(f"  [shared] step={self.num_timesteps:,} | "
                      f"checkpoint saved → {path}.zip")

        return True


# SETUP VECTORIZED ENV (4 robot = 4 env)
def make_env(robot_id: int, seed: int):
    """Factory function untuk satu env — diperlukan oleh VecEnv."""
    def _init():
        env = GazeboFormationEnv(robot_id=robot_id, seed=seed + robot_id)
        return env
    return _init


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 60)
    print("Parameter-Sharing PPO V2 — Square + Smoothness Penalty")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"Buffer per update: {PPO_PARAMS['n_steps']} × 4 robot = "
          f"{PPO_PARAMS['n_steps'] * 4:,}")
    print(f"Models dir      : {MODELS_DIR}")
    print()
    print("          sudah berjalan di terminal lain!")
    print()
    input("Tekan ENTER untuk mulai training...")

    base_seed = int(time.time())
    print(f"Base seed: {base_seed}\n")

    # Vectorized Environment
    print("Inisialisasi 4 environment...")
    vec_env = DummyVecEnv([
        make_env(robot_id=i, seed=base_seed)
        for i in range(len(ROBOT_NAMES))
    ])
    vec_env = VecMonitor(vec_env) # wajib agar key "episode" tersedia di info
    print("4 environment siap (dengan VecMonitor).\n")

    # Shared PPO Model
    CHECKPOINT = "ppo_square_smooth_final" # Fresh start untuk eksperimen smoothness

    if CHECKPOINT is not None:
        checkpoint_path = os.path.join(MODELS_DIR, CHECKPOINT)
        if os.path.exists(checkpoint_path + ".zip"):
            print(f"Melanjutkan dari checkpoint: {checkpoint_path}.zip")
            model = PPO.load(
                checkpoint_path,
                env             = vec_env,
                tensorboard_log = os.path.join(LOGS_DIR, "ppo_square_smooth"),
            )
        else:
            print(f"WARNING: Checkpoint {checkpoint_path}.zip tidak ditemukan!")
            print("Fallback ke fresh start.")
            model = PPO(
                policy          = "MlpPolicy",
                env             = vec_env,
                tensorboard_log = os.path.join(LOGS_DIR, "ppo_square_smooth"),
                policy_kwargs   = POLICY_KWARGS,
                **PPO_PARAMS
            )
    else:
        print("Fresh start training.")
        model = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            tensorboard_log = os.path.join(LOGS_DIR, "ppo_square_smooth"),
            policy_kwargs   = POLICY_KWARGS,
            **PPO_PARAMS
        )

    print("Arsitektur policy network:")
    print(f"  Input  : 8 dimensi [dx1,dy1,dx2,dy2,v,w,dpsi1,dpsi2]")
    print(f"  Hidden : 64 → 64")
    print(f"  Output : 2 dimensi [v, w] (continuous)")
    print()

    # Training
    callback = SharedPPOCallback(
        save_dir   = MODELS_DIR,
        save_every = SAVE_EVERY_STEPS,
    )

    t_start = time.time()
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = callback,
        log_interval        = LOG_INTERVAL,
        reset_num_timesteps = (CHECKPOINT is None),  # False jika dari checkpoint
        tb_log_name         = "ppo_square_smooth",
    )
    t_elapsed = time.time() - t_start

    # Simpan Model Final
    final_path = os.path.join(MODELS_DIR, "ppo_square_smooth_final")
    model.save(final_path)

    print(f"\n{'='*60}")
    print(f"Training selesai dalam {t_elapsed/60:.1f} menit")
    print(f"Model final: {final_path}.zip")
    print(f"TensorBoard: tensorboard --logdir {LOGS_DIR}")
    print(f"{'='*60}")

    vec_env.close()


#  ENTRY POINT
if __name__ == "__main__":
    train()
