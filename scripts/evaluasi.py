#!/usr/bin/env python3
from __future__ import annotations
"""
evaluasi.py
===========
Script evaluasi formal untuk formasi 4 robot.

Menjalankan N episode tanpa training, mencatat metrik per episode:
- mean_distance_error  : rata-rata |d_actual - D_TARGET| ke neighbor terdekat
- mean_heading_error   : rata-rata |dpsi1| dalam derajat
- collinearity_error   : rata-rata jarak keempat robot dari garis lurus terbaik
- collision_count      : jumlah timestep dengan jarak < D_SAFE
- formation_steps      : timestep pertama semua metrik memenuhi threshold (opsional)

Cara pakai:
    python3 evaluasi.py --model models/ppo_shared_step450012 --episodes 20
    python3 evaluasi.py --model models/ppo_shared_step310012 --episodes 20
"""

import argparse
import os
import sys
import math
import numpy as np
import rospy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_gazebo import GazeboFormationEnv, ROBOT_NAMES, _shared_state, D_TARGET, D_SAFE

import time


# ─────────────────────────────────────────────
#  HELPER: COLLINEARITY ERROR
# ─────────────────────────────────────────────

def compute_collinearity_error() -> float:
    """
    Hitung rata-rata jarak keempat robot dari garis lurus terbaik (least squares fit).
    
    Metode:
    1. Ambil posisi (x, y) keempat robot dari shared state
    2. Fit garis lurus menggunakan least squares (numpy.polyfit)
    3. Hitung jarak tegak lurus tiap robot ke garis tersebut
    4. Return rata-rata jarak
    
    Nilai mendekati 0 = keempat robot hampir sempurna membentuk garis lurus.
    """
    positions = [(
        _shared_state[name]["x"],
        _shared_state[name]["y"]
    ) for name in ROBOT_NAMES]

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    # Cek apakah semua robot hampir di titik yang sama (degenerate case)
    spread = np.std(xs) + np.std(ys)
    if spread < 0.01:
        return 0.0

    # Fit garis: pilih orientasi fit berdasarkan variance
    # Kalau xs lebih tersebar → fit y = ax + b
    # Kalau ys lebih tersebar → fit x = ay + b (hindari vertical line problem)
    if np.std(xs) >= np.std(ys):
        coeffs = np.polyfit(xs, ys, 1)   # y = ax + b
        a, b = coeffs
        # Jarak titik (x0,y0) ke garis ax - y + b = 0: |ax0 - y0 + b| / sqrt(a²+1)
        dists = [abs(a * x - y + b) / math.sqrt(a**2 + 1)
                 for x, y in positions]
    else:
        coeffs = np.polyfit(ys, xs, 1)   # x = ay + b
        a, b = coeffs
        dists = [abs(a * y - x + b) / math.sqrt(a**2 + 1)
                 for x, y in positions]

    return float(np.mean(dists))


def compute_heading_error(env: GazeboFormationEnv) -> float:
    """
    Rata-rata |dpsi1| dalam derajat untuk robot yang bersangkutan.
    Dihitung dari shared state langsung.
    """
    me = _shared_state[env.robot_name]
    others = sorted(
        [(math.hypot(_shared_state[n]["x"] - me["x"],
                     _shared_state[n]["y"] - me["y"]), n)
         for n in ROBOT_NAMES if n != env.robot_name]
    )
    n1_yaw = _shared_state[others[0][1]]["yaw"]
    dpsi   = abs(env._wrap_angle(n1_yaw - me["yaw"]))
    return math.degrees(dpsi)


def compute_distance_error(env: GazeboFormationEnv) -> float:
    """
    |d_actual - D_TARGET| ke neighbor terdekat robot ini.
    """
    me = _shared_state[env.robot_name]
    others = sorted(
        [(math.hypot(_shared_state[n]["x"] - me["x"],
                     _shared_state[n]["y"] - me["y"]), n)
         for n in ROBOT_NAMES if n != env.robot_name]
    )
    d1 = others[0][0]
    return abs(d1 - D_TARGET)


def check_any_collision() -> bool:
    """Cek apakah ada pasangan robot yang jaraknya < D_SAFE."""
    names = ROBOT_NAMES
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = math.hypot(
                _shared_state[names[i]]["x"] - _shared_state[names[j]]["x"],
                _shared_state[names[i]]["y"] - _shared_state[names[j]]["y"]
            )
            if d < D_SAFE:
                return True
    return False


# ─────────────────────────────────────────────
#  EVALUASI SATU EPISODE
# ─────────────────────────────────────────────

def evaluate_episode(envs: list, model: PPO, episode_num: int) -> dict:
    """
    Jalankan satu episode evaluasi, return dict metrik.
    """
    # Reset semua env
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    obs_array = np.array(obs_list)

    # Akumulasi metrik per step
    step_distance_errors  = []
    step_heading_errors   = []
    step_collinearity     = []
    collision_count       = 0

    done = False
    step = 0

    while not done:
        # Predict action untuk semua robot sekaligus
        actions, _ = model.predict(obs_array, deterministic=True)

        # Step semua env
        new_obs_list = []
        for i, env in enumerate(envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            new_obs_list.append(obs)
            if truncated:
                done = True

        obs_array = np.array(new_obs_list)
        step += 1

        # Catat metrik per step
        dist_errors = [compute_distance_error(env) for env in envs]
        hdg_errors  = [compute_heading_error(env) for env in envs]
        coll_err    = compute_collinearity_error()

        step_distance_errors.append(np.mean(dist_errors))
        step_heading_errors.append(np.mean(hdg_errors))
        step_collinearity.append(coll_err)

        if check_any_collision():
            collision_count += 1

    # Ambil metrik di 100 step terakhir (kondisi akhir episode lebih relevan)
    last_n = 100
    final_distance_error  = float(np.mean(step_distance_errors[-last_n:]))
    final_heading_error   = float(np.mean(step_heading_errors[-last_n:]))
    final_collinearity    = float(np.mean(step_collinearity[-last_n:]))

    result = {
        "episode"               : episode_num,
        "mean_distance_error_m" : round(final_distance_error, 4),
        "mean_heading_error_deg": round(final_heading_error, 2),
        "collinearity_error_m"  : round(final_collinearity, 4),
        "collision_count"       : collision_count,
        "total_steps"           : step,
    }

    print(f"\nEpisode {episode_num:02d}:")
    print(f"  distance_error  = {result['mean_distance_error_m']:.4f} m")
    print(f"  heading_error   = {result['mean_heading_error_deg']:.2f} °")
    print(f"  collinearity    = {result['collinearity_error_m']:.4f} m")
    print(f"  collision_count = {result['collision_count']} steps")

    return result


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluasi formal formasi 4 robot")
    parser.add_argument("--model",    type=str,
                        default="models/ppo_shared_step450012",
                        help="Path ke model checkpoint (tanpa .zip)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Jumlah episode evaluasi")
    args = parser.parse_args()

    # Resolve path relatif ke lokasi script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", args.model)
    model_path = os.path.normpath(model_path)

    print("=" * 60)
    print("Evaluasi Formal — Formasi 4 Robot")
    print("=" * 60)
    print(f"Model    : {model_path}.zip")
    print(f"Episodes : {args.episodes}")
    print()
    print("PASTIKAN: roslaunch marl_formation formasi_4_robot.launch")
    print("          sudah berjalan di terminal lain!")
    print()
    input("Tekan ENTER untuk mulai evaluasi...")

    # Inisialisasi 4 env (satu per robot)
    print("\nInisialisasi environment...")
    envs = [GazeboFormationEnv(robot_id=i, seed=42 + i)
            for i in range(len(ROBOT_NAMES))]
    print("4 environment siap.\n")

    # Load model
    # Buat dummy vec env hanya untuk load (model.predict tidak butuh vec env)
    dummy_env = DummyVecEnv([lambda: GazeboFormationEnv(robot_id=0, seed=99)])
    model = PPO.load(model_path, env=dummy_env)
    dummy_env.close()
    print(f"Model loaded: {model_path}.zip\n")

    # Jalankan evaluasi
    all_results = []
    for ep in range(1, args.episodes + 1):
        result = evaluate_episode(envs, model, ep)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY EVALUASI")
    print("=" * 60)

    dist_errors  = [r["mean_distance_error_m"]  for r in all_results]
    hdg_errors   = [r["mean_heading_error_deg"] for r in all_results]
    coll_errors  = [r["collinearity_error_m"]   for r in all_results]
    collisions   = [r["collision_count"]        for r in all_results]

    print(f"Distance error  : mean={np.mean(dist_errors):.4f}m  "
          f"std={np.std(dist_errors):.4f}m  "
          f"min={np.min(dist_errors):.4f}m  max={np.max(dist_errors):.4f}m")
    print(f"Heading error   : mean={np.mean(hdg_errors):.2f}°  "
          f"std={np.std(hdg_errors):.2f}°  "
          f"min={np.min(hdg_errors):.2f}°  max={np.max(hdg_errors):.2f}°")
    print(f"Collinearity    : mean={np.mean(coll_errors):.4f}m  "
          f"std={np.std(coll_errors):.4f}m  "
          f"min={np.min(coll_errors):.4f}m  max={np.max(coll_errors):.4f}m")
    print(f"Collision steps : mean={np.mean(collisions):.1f}  "
          f"total={sum(collisions)}  "
          f"max={max(collisions)}")
    print()
    print("─" * 60)
    print("Distribusi (untuk menentukan threshold success):")
    print(f"  distance_error < 0.3m : "
          f"{sum(1 for x in dist_errors if x < 0.3)}/{len(dist_errors)} episode")
    print(f"  distance_error < 0.5m : "
          f"{sum(1 for x in dist_errors if x < 0.5)}/{len(dist_errors)} episode")
    print(f"  heading_error  < 20°  : "
          f"{sum(1 for x in hdg_errors if x < 20)}/{len(hdg_errors)} episode")
    print(f"  heading_error  < 30°  : "
          f"{sum(1 for x in hdg_errors if x < 30)}/{len(hdg_errors)} episode")
    print(f"  collinearity   < 0.3m : "
          f"{sum(1 for x in coll_errors if x < 0.3)}/{len(coll_errors)} episode")
    print(f"  collinearity   < 0.5m : "
          f"{sum(1 for x in coll_errors if x < 0.5)}/{len(coll_errors)} episode")
    print(f"  zero collision        : "
          f"{sum(1 for x in collisions if x == 0)}/{len(collisions)} episode")
    print("─" * 60)
    print()
    print("→ Gunakan distribusi di atas untuk menentukan threshold success.")
    print("  Jangan tentukan threshold sebelum melihat data ini.")

    # Simpan hasil ke file
    output_path = os.path.join(script_dir, "..", "logs", "evaluasi_results.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {args.episodes}\n\n")
        for r in all_results:
            f.write(str(r) + "\n")
        f.write(f"\nMean distance_error : {np.mean(dist_errors):.4f}\n")
        f.write(f"Mean heading_error  : {np.mean(hdg_errors):.2f}\n")
        f.write(f"Mean collinearity   : {np.mean(coll_errors):.4f}\n")
        f.write(f"Mean collision_steps: {np.mean(collisions):.1f}\n")
    print(f"\nHasil disimpan: {output_path}")

    # Cleanup
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
