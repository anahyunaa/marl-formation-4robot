#!/usr/bin/env python3
from __future__ import annotations
"""
diagnosis_angle.py
==================
Diagnosis per-robot reward dan sudut formasi di akhir episode.
Jalankan dengan seed berbeda untuk verifikasi konsistensi.

Cara pakai:
    python3 diagnosis_angle.py          # seed default (42)
    python3 diagnosis_angle.py --seed 100
    python3 diagnosis_angle.py --seed 200
"""

import sys
import math
import argparse
import numpy as np

sys.path.insert(0, '.')
from env_gazebo import GazeboFormationEnv, ROBOT_NAMES, _shared_state, D_TARGET

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import rospy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--model", type=str,
                        default="../models/ppo_square_step200000")
    parser.add_argument("--steps", type=int, default=700)
    args = parser.parse_args()

    # Load model
    dummy = DummyVecEnv([lambda: GazeboFormationEnv(robot_id=0, seed=99)])
    model = PPO.load(args.model, env=dummy)
    dummy.close()

    # Init envs
    envs = [GazeboFormationEnv(robot_id=i, seed=args.seed + i)
            for i in range(len(ROBOT_NAMES))]

    # Reset
    obs_list = [env.reset()[0] for env in envs]
    obs_array = np.array(obs_list)

    # Print posisi spawn awal
    rospy.sleep(0.5)
    print(f"\n=== Seed: {args.seed} ===")
    print("Posisi Spawn Awal:")
    for name in ROBOT_NAMES:
        s = _shared_state[name]
        print(f"  {name}: x={s['x']:.3f}, y={s['y']:.3f}")

    # Run episode + catat reward per robot
    robot_rewards = {name: [] for name in ROBOT_NAMES}

    for step in range(args.steps):
        actions, _ = model.predict(obs_array, deterministic=True)
        new_obs = []
        for i, env in enumerate(envs):
            obs, reward, _, _, _ = env.step(actions[i])
            new_obs.append(obs)
            robot_rewards[ROBOT_NAMES[i]].append(reward)
        obs_array = np.array(new_obs)

    # Print posisi final
    print("\nPosisi Final:")
    for name in ROBOT_NAMES:
        s = _shared_state[name]
        print(f"  {name}: x={s['x']:.3f}, y={s['y']:.3f}, "
              f"yaw={math.degrees(s['yaw']):.1f}°")

    # Print sudut per robot
    print("\nSudut yang Dilihat Tiap Robot:")
    for name in ROBOT_NAMES:
        me     = _shared_state[name]
        others = sorted([
            (math.hypot(_shared_state[n]["x"] - me["x"],
                        _shared_state[n]["y"] - me["y"]), n)
            for n in ROBOT_NAMES if n != name
        ])
        n1 = _shared_state[others[0][1]]
        n2 = _shared_state[others[1][1]]
        d1, d2 = others[0][0], others[1][0]

        yaw  = me["yaw"]
        dx1  =  (n1["x"]-me["x"])*math.cos(yaw) + (n1["y"]-me["y"])*math.sin(yaw)
        dy1  = -(n1["x"]-me["x"])*math.sin(yaw) + (n1["y"]-me["y"])*math.cos(yaw)
        dx2  =  (n2["x"]-me["x"])*math.cos(yaw) + (n2["y"]-me["y"])*math.sin(yaw)
        dy2  = -(n2["x"]-me["x"])*math.sin(yaw) + (n2["y"]-me["y"])*math.cos(yaw)

        cos_a = (dx1*dx2 + dy1*dy2) / (d1*d2 + 1e-6)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.degrees(math.acos(cos_a))

        print(f"  {name}: neighbor=({others[0][1]},{others[1][1]})  "
              f"d1={d1:.2f}m  d2={d2:.2f}m  sudut={angle:.1f}°")

    # Print reward per robot (100 step terakhir)
    print("\nMean Reward 100 Step Terakhir per Robot:")
    for name in ROBOT_NAMES:
        last100 = robot_rewards[name][-100:]
        print(f"  {name}: mean={np.mean(last100):.4f}  "
              f"min={np.min(last100):.4f}  "
              f"max={np.max(last100):.4f}")

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
