#!/usr/bin/env python3
from __future__ import annotations
"""
evaluasi_square.py
==================
Script evaluasi formal untuk square formation 4 robot.

Metrik per episode:
- side_distance_error : rata-rata |d_actual - D_TARGET| ke 2 neighbor
- angle_error_deg     : seberapa jauh sudut antar 2 neighbor dari 90°
- collision_count     : jumlah timestep dengan jarak < D_SAFE
- heading_error_deg   : rata-rata |dpsi| ke 2 neighbor (secondary)

Cara pakai:
    python3 evaluasi_square.py --model models/ppo_square_step160000 --episodes 20
    python3 evaluasi_square.py --model models/ppo_square_step100000 --episodes 20
    python3 evaluasi_square.py --model models/ppo_square_step200000 --episodes 20
"""

import argparse
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_gazebo import (GazeboFormationEnv, ROBOT_NAMES,
                        _shared_state, D_TARGET, D_SAFE)

import rospy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ─────────────────────────────────────────────
#  FORMATION SUCCESS THRESHOLD (tunable)
# ─────────────────────────────────────────────
FORM_ANGLE_THRESH = 30.0   # derajat — angle error harus di bawah ini
FORM_DIST_THRESH  = 1.0    # meter   — dist error harus di bawah ini
FORM_WINDOW       = 50     # step berturutan yang harus memenuhi threshold


# ─────────────────────────────────────────────
#  HELPER METRICS
# ─────────────────────────────────────────────

def get_sorted_neighbors(robot_name: str) -> list:
    me = _shared_state[robot_name]
    others = []
    for name in ROBOT_NAMES:
        if name == robot_name:
            continue
        d = math.hypot(
            _shared_state[name]["x"] - me["x"],
            _shared_state[name]["y"] - me["y"]
        )
        others.append((d, name))
    others.sort(key=lambda t: t[0])
    return others


def to_body_frame(me: dict, other: dict) -> tuple:
    dx_w = other["x"] - me["x"]
    dy_w = other["y"] - me["y"]
    yaw  = me["yaw"]
    dx_l =  dx_w * math.cos(yaw) + dy_w * math.sin(yaw)
    dy_l = -dx_w * math.sin(yaw) + dy_w * math.cos(yaw)
    return dx_l, dy_l


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_metrics_all_robots() -> dict:
    """
    Hitung metrik untuk semua robot, return rata-rata.
    """
    dist_errors  = []
    angle_errors = []
    hdg_errors   = []

    for robot_name in ROBOT_NAMES:
        me     = _shared_state[robot_name]
        others = get_sorted_neighbors(robot_name)

        n1 = _shared_state[others[0][1]]
        n2 = _shared_state[others[1][1]]
        d1 = others[0][0]
        d2 = others[1][0]

        dx1, dy1 = to_body_frame(me, n1)
        dx2, dy2 = to_body_frame(me, n2)

        # Side distance error — rata-rata error ke 2 neighbor
        dist_err = (abs(d1 - D_TARGET) + abs(d2 - D_TARGET)) / 2.0
        dist_errors.append(dist_err)

        # Angle error — sudut antar 2 neighbor vs target 90°
        cos_angle   = (dx1*dx2 + dy1*dy2) / (d1 * d2 + 1e-6)
        cos_angle   = max(-1.0, min(1.0, cos_angle))  # clamp numerical errors
        actual_deg  = math.degrees(math.acos(cos_angle))  # tanpa abs() — 60° ≠ 120°
        # Target 90°: error = |actual - 90|
        angle_err   = abs(actual_deg - 90.0)
        angle_errors.append(angle_err)

        # Heading error ke 2 neighbor
        dpsi1 = abs(wrap_angle(n1["yaw"] - me["yaw"]))
        dpsi2 = abs(wrap_angle(n2["yaw"] - me["yaw"]))
        hdg_errors.append(math.degrees((dpsi1 + dpsi2) / 2.0))

    return {
        "dist_error" : float(np.mean(dist_errors)),
        "angle_error": float(np.mean(angle_errors)),
        "hdg_error"  : float(np.mean(hdg_errors)),
    }


def check_any_collision() -> bool:
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
    # Reset
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    obs_array = np.array(obs_list)

    step_dist   = []
    step_angle  = []
    step_hdg    = []
    collision_count = 0
    done = False

    # Formation tracking (threshold defined at module level)
    formation_achieved    = False
    time_to_formation     = -1   # -1 = tidak tercapai
    consecutive_form_steps = 0
    step_counter          = 0

    while not done:
        actions, _ = model.predict(obs_array, deterministic=True)

        new_obs_list = []
        for i, env in enumerate(envs):
            obs, _, _, truncated, _ = env.step(actions[i])
            new_obs_list.append(obs)
            if truncated:
                done = True

        obs_array = np.array(new_obs_list)

        m = compute_metrics_all_robots()
        step_dist.append(m["dist_error"])
        step_angle.append(m["angle_error"])
        step_hdg.append(m["hdg_error"])

        if check_any_collision():
            collision_count += 1

        # Cek apakah formasi tercapai di step ini
        step_counter += 1
        m_now = compute_metrics_all_robots()
        if (m_now["angle_error"] < FORM_ANGLE_THRESH and
                m_now["dist_error"] < FORM_DIST_THRESH):
            consecutive_form_steps += 1
            if (consecutive_form_steps >= FORM_WINDOW
                    and not formation_achieved):
                formation_achieved = True
                time_to_formation  = step_counter
        else:
            consecutive_form_steps = 0  # reset kalau keluar threshold

    # Rata-rata 100 step terakhir
    last_n = 100
    result = {
        "episode"                  : episode_num,
        # Mean 100 step terakhir
        "dist_error_m"             : round(float(np.mean(step_dist[-last_n:])), 4),
        "angle_error_deg"          : round(float(np.mean(step_angle[-last_n:])), 2),
        "heading_error_deg"        : round(float(np.mean(step_hdg[-last_n:])), 2),
        # Kondisi step terakhir (untuk pertanyaan sidang "kondisi akhir formasi")
        "final_dist_error_m"       : round(float(step_dist[-1]), 4),
        "final_angle_error_deg"    : round(float(step_angle[-1]), 2),
        "collision_count"          : collision_count,
        "formation_achieved"       : formation_achieved,
        "time_to_formation"        : time_to_formation,
    }

    print(f"\nEpisode {episode_num:02d}:")
    print(f"  dist_error   = {result['dist_error_m']:.4f} m  "
          f"(final: {result['final_dist_error_m']:.4f} m)")
    print(f"  angle_error  = {result['angle_error_deg']:.2f} °  "
          f"(final: {result['final_angle_error_deg']:.2f} °)")
    print(f"  heading_err  = {result['heading_error_deg']:.2f} °")
    print(f"  collisions   = {result['collision_count']} steps")
    status = f"step {result['time_to_formation']}" if result['formation_achieved'] else "TIDAK TERCAPAI"
    print(f"  formation    = {'✓ TERCAPAI' if result['formation_achieved'] else '✗ tidak'} ({status})")

    return result


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,
                        default="models/ppo_square_step160000",
                        help="Path ke model (tanpa .zip)")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(
        os.path.join(script_dir, "..", args.model)
    )

    print("=" * 60)
    print("Evaluasi Square Formation — 4 Robot")
    print("=" * 60)
    print(f"Model    : {model_path}.zip")
    print(f"Episodes : {args.episodes}")
    print()
    print("PASTIKAN: roslaunch marl_formation formasi_4_robot.launch")
    print("          sudah berjalan!")
    print()
    input("Tekan ENTER untuk mulai...")

    # Inisialisasi env
    print("\nInisialisasi 4 environment...")
    envs = [GazeboFormationEnv(robot_id=i, seed=42 + i)
            for i in range(len(ROBOT_NAMES))]

    # Load model
    dummy = DummyVecEnv([lambda: GazeboFormationEnv(robot_id=0, seed=99)])
    model = PPO.load(model_path, env=dummy)
    dummy.close()
    print(f"Model loaded.\n")

    # Evaluasi
    all_results = []
    for ep in range(1, args.episodes + 1):
        result = evaluate_episode(envs, model, ep)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY EVALUASI — SQUARE FORMATION")
    print("=" * 60)

    dist_errs       = [r["dist_error_m"]         for r in all_results]
    angle_errs      = [r["angle_error_deg"]      for r in all_results]
    hdg_errs        = [r["heading_error_deg"]    for r in all_results]
    collisions      = [r["collision_count"]      for r in all_results]
    final_dist_errs = [r["final_dist_error_m"]   for r in all_results]
    final_ang_errs  = [r["final_angle_error_deg"] for r in all_results]

    # Urutan: Distance → Angle → Collision → Heading (sesuai prioritas reward)
    print(f"1. Side distance error (mean 100 last steps):")
    print(f"   mean={np.mean(dist_errs):.4f}m  std={np.std(dist_errs):.4f}m  "
          f"min={np.min(dist_errs):.4f}m  max={np.max(dist_errs):.4f}m")
    print(f"   final-step: mean={np.mean(final_dist_errs):.4f}m")
    print(f"2. Angle error vs 90° (mean 100 last steps):")
    print(f"   mean={np.mean(angle_errs):.2f}°  std={np.std(angle_errs):.2f}°  "
          f"min={np.min(angle_errs):.2f}°  max={np.max(angle_errs):.2f}°")
    print(f"   final-step: mean={np.mean(final_ang_errs):.2f}°")
    print(f"3. Collision steps:")
    print(f"   mean={np.mean(collisions):.1f}  total={sum(collisions)}  "
          f"max={max(collisions)}")
    print(f"4. Heading error (secondary):")
    print(f"   mean={np.mean(hdg_errs):.2f}°  std={np.std(hdg_errs):.2f}°")

    # Formation success metrics
    form_achieved  = [r["formation_achieved"] for r in all_results]
    form_times     = [r["time_to_formation"]  for r in all_results
                      if r["formation_achieved"]]
    success_count  = sum(form_achieved)
    print(f"\n5. Formation Success (angle<{FORM_ANGLE_THRESH:.0f}° AND dist<{FORM_DIST_THRESH:.1f}m selama {FORM_WINDOW} step berturutan):")
    print(f"   success rate : {success_count}/{len(all_results)} episode  ({100*success_count/len(all_results):.0f}%)")
    if form_times:
        print(f"   time to form : mean={np.mean(form_times):.0f} step  "
              f"min={np.min(form_times)} step  max={np.max(form_times)} step")
    else:
        print(f"   time to form : N/A (tidak ada episode yang berhasil)")

    print()
    print("─" * 60)
    print("Distribusi — dasar penentuan threshold success:")
    thresholds = [
        ("dist_error  < 0.3m",  dist_errs,  lambda x: x < 0.3),
        ("dist_error  < 0.5m",  dist_errs,  lambda x: x < 0.5),
        ("angle_error < 10°",   angle_errs, lambda x: x < 10),
        ("angle_error < 20°",   angle_errs, lambda x: x < 20),
        ("angle_error < 30°",   angle_errs, lambda x: x < 30),
        ("heading_err < 20°",   hdg_errs,   lambda x: x < 20),
        ("heading_err < 30°",   hdg_errs,   lambda x: x < 30),
        ("zero collision",      collisions, lambda x: x == 0),
    ]
    for label, data, cond in thresholds:
        count = sum(1 for x in data if cond(x))
        print(f"  {label:22s}: {count:2d}/{len(data)} episode")
    print("─" * 60)

    # Simpan hasil
    out_path = os.path.join(script_dir, "..", "logs",
                            f"eval_square_{args.model.split('/')[-1]}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Model: {model_path}\n\n")
        for r in all_results:
            f.write(str(r) + "\n")
        f.write(f"\nmean_dist_error  : {np.mean(dist_errs):.4f}\n")
        f.write(f"mean_angle_error : {np.mean(angle_errs):.2f}\n")
        f.write(f"mean_heading_err : {np.mean(hdg_errs):.2f}\n")
        f.write(f"mean_collisions  : {np.mean(collisions):.1f}\n")
    print(f"\nHasil disimpan: {out_path}")

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
