#!/usr/bin/env python3
"""
evaluate_demo.py
================
Script demo untuk keperluan presentasi sidang.
Episode BERHENTI segera saat formasi tercapai — robot diam di posisi formasi.

METODOLOGI PENELITIAN TIDAK BERUBAH.
File ini hanya untuk kebutuhan video demo presentasi.

Cara pakai:
    python3 evaluate_demo.py
    python3 evaluate_demo.py --model models/baseline/ppo_square_step180000
    python3 evaluate_demo.py --seed 7
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
from geometry_msgs.msg import Twist
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Threshold demo — lebih ketat dari evaluasi.py
# Tujuan: robot berhenti hanya saat formasi benar-benar terlihat bagus secara visual
# evaluasi.py tetap pakai: dist<1.0m, angle<30°, window=50
FORM_ANGLE_THRESH = 15.0  # lebih ketat dari 30°
FORM_DIST_THRESH  = 0.6   # lebih ketat dari 1.0m
FORM_WINDOW       = 5     # cukup untuk demo visual
MIN_STEPS_BEFORE_CHECK = 50   # Jangan cek formasi sebelum robot sempat bergerak


def get_sorted_neighbors(robot_name):
    me = _shared_state[robot_name]
    others = [
        (math.hypot(_shared_state[n]["x"] - me["x"],
                    _shared_state[n]["y"] - me["y"]), n)
        for n in ROBOT_NAMES if n != robot_name
    ]
    others.sort(key=lambda t: t[0])
    return others


def to_body_frame(me, other):
    dx_w = other["x"] - me["x"]
    dy_w = other["y"] - me["y"]
    yaw  = me["yaw"]
    return (dx_w * math.cos(yaw) + dy_w * math.sin(yaw),
           -dx_w * math.sin(yaw) + dy_w * math.cos(yaw))


def compute_metrics():
    dist_errors, angle_errors = [], []
    for name in ROBOT_NAMES:
        me     = _shared_state[name]
        others = get_sorted_neighbors(name)
        d1, d2 = others[0][0], others[1][0]
        n1, n2 = _shared_state[others[0][1]], _shared_state[others[1][1]]
        dx1, dy1 = to_body_frame(me, n1)
        dx2, dy2 = to_body_frame(me, n2)

        dist_errors.append((abs(d1 - D_TARGET) + abs(d2 - D_TARGET)) / 2.0)

        cos_a = (dx1*dx2 + dy1*dy2) / (d1 * d2 + 1e-6)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle_errors.append(abs(math.degrees(math.acos(cos_a)) - 90.0))

    return float(np.mean(dist_errors)), float(np.mean(angle_errors))


def stop_all_robots(publishers):
    for pub in publishers:
        pub.publish(Twist())
    rospy.sleep(0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/smooth_v2/ppo_square_smooth_step200000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(script_dir, "..", args.model))

    print("=" * 50)
    print("  DEMO FORMASI")
    print(f"  Model : {args.model.split('/')[-1]}")
    print("=" * 50)
    print()
    input("Tekan ENTER untuk mulai...")

    print("\nMemuat model...")
    envs = [GazeboFormationEnv(robot_id=i, seed=args.seed + i)
            for i in range(len(ROBOT_NAMES))]

    cmd_publishers = [
        rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)
        for name in ROBOT_NAMES
    ]

    dummy = DummyVecEnv([lambda: GazeboFormationEnv(robot_id=0, seed=99)])
    model = PPO.load(model_path, env=dummy)
    dummy.close()

    obs_list = [env.reset()[0] for env in envs]
    obs_array = np.array(obs_list)

    print("Menjalankan demo...")
    step_counter           = 0
    formation_achieved     = False
    consecutive_form_steps = 0

    while True:
        actions, _ = model.predict(obs_array, deterministic=True)

        new_obs_list = []
        truncated_any = False
        for i, env in enumerate(envs):
            obs, _, _, truncated, _ = env.step(actions[i])
            new_obs_list.append(obs)
            if truncated:
                truncated_any = True

        obs_array    = np.array(new_obs_list)
        step_counter += 1

        dist_err, angle_err = compute_metrics()

        # Cek formasi — hanya setelah MIN_STEPS_BEFORE_CHECK
        # agar robot sempat bergerak dulu sebelum dinyatakan tercapai
        if (step_counter >= MIN_STEPS_BEFORE_CHECK and
                dist_err < FORM_DIST_THRESH and angle_err < FORM_ANGLE_THRESH):
            consecutive_form_steps += 1
            if consecutive_form_steps >= FORM_WINDOW:
                formation_achieved = True
                break
        else:
            consecutive_form_steps = 0

        if truncated_any:
            break

    # Hentikan robot
    print("\nMenghentikan robot...")
    stop_all_robots(cmd_publishers)

    # Hasil
    print()
    print("=" * 50)
    if formation_achieved:
        dist_final, angle_final = compute_metrics()
        print("  FORMASI TERCAPAI")
        print(f"  Distance error : {dist_final:.3f} m")
        print(f"  Angle error    : {angle_final:.2f}°")
        print(f"  Waktu          : {step_counter} step")
        print()
        print("  Robot berhenti.")
    else:
        print("  ✗ Formasi tidak tercapai (750 step habis)")
        print("  Coba jalankan ulang dengan --seed berbeda")
    print("=" * 50)

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
