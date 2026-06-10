#!/usr/bin/env python3
from __future__ import annotations
"""
env_gazebo.py — Square Formation
=================================
Gymnasium environment wrapper untuk IPPO training formasi persegi 4 robot.

Target formasi:
- 4 robot membentuk persegi dengan sisi D_TARGET = 1.5m
- Sudut antar neighbor = 90°
- Heading seragam (relatif, bukan absolut)

Reward components:
- r_dist1, r_dist2 : spacing ke 2 neighbor terdekat → D_TARGET
- r_angle          : sudut antara 2 neighbor → 90° (cos → 0)
- r_hdg            : heading alignment ke 2 neighbor → dpsi → 0
- r_col            : collision penalty

Changelog dari line formation:
- Reward total diganti untuk square formation
- MAX_STEPS: 1000 → 750
- SPAWN_HALF: 2.0 → 1.5
- GAMMA tetap 1.0 tapi r_hdg diberi bobot 0.5 di reward
- Training fresh start (tidak dari checkpoint)
"""

import rospy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import math
import time

# ─────────────────────────────────────────────
#  KONSTANTA GLOBAL
# ─────────────────────────────────────────────

ROBOT_NAMES     = ["r1", "r2", "r3", "r4"]
N_ROBOTS        = len(ROBOT_NAMES)

CONTROL_RATE    = 10        # Hz
MAX_STEPS       = 750       # turun dari 1000

V_MAX           = 0.3       # m/s
W_MAX           = 0.5       # rad/s

D_TARGET        = 1.5       # meter — sisi persegi
D_SAFE          = 0.75      # meter — collision threshold

# Reward weights
ALPHA           = 1.0       # bobot r_dist (per neighbor)
GAMMA           = 1.0       # bobot r_hdg (di reward pakai 0.5 * GAMMA)
ANGLE_W         = 5.0       # bobot r_angle — dinaikkan untuk dorong 90°
COLLISION_PEN   = 20.0      # penalty collision

# Spawn
SPAWN_HALF      = 1.5       # turun dari 2.0
SPAWN_MIN_DIST  = 1.0

# Smoothness penalty (jerk penalty)
LAMBDA_SMOOTH   = 0.1    # bobot penalty perubahan action antar timestep

# Curriculum Learning
RANDOM_YAW      = False     # Phase 1: yaw=0, Phase 2: True


# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────

_shared_state = {
    name: {"x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0, "w": 0.0}
    for name in ROBOT_NAMES
}
_ros_initialized = False


def _ensure_ros_init():
    global _ros_initialized
    if not _ros_initialized:
        rospy.init_node("marl_formation_env", anonymous=False,
                        disable_signals=True)
        _ros_initialized = True


def _odom_callback(msg: Odometry, robot_name: str):
    x   = msg.pose.pose.position.x
    y   = msg.pose.pose.position.y
    q   = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    v   = msg.twist.twist.linear.x
    w   = msg.twist.twist.angular.z
    _shared_state[robot_name].update(
        {"x": x, "y": y, "yaw": yaw, "v": v, "w": w}
    )


def _random_spawn_positions(n: int, half: float, min_dist: float,
                             rng: np.random.Generator) -> list:
    positions = []
    for _ in range(1000):
        candidate = (
            rng.uniform(-half, half),
            rng.uniform(-half, half),
            rng.uniform(-math.pi, math.pi) if RANDOM_YAW else 0.0,
        )
        too_close = any(
            math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < min_dist
            for p in positions
        )
        if not too_close:
            positions.append(candidate)
        if len(positions) == n:
            return positions

    rospy.logwarn("spawn: fallback ke grid")
    grid = [(-0.75, -0.75), (-0.75, 0.75), (0.75, -0.75), (0.75, 0.75)]
    return [(gx + rng.uniform(-0.1, 0.1),
             gy + rng.uniform(-0.1, 0.1),
             0.0) for gx, gy in grid[:n]]


# ─────────────────────────────────────────────
#  ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class GazeboFormationEnv(gym.Env):
    """
    Square formation environment untuk parameter-sharing PPO.

    State: [dx1, dy1, dx2, dy2, v_self, w_self, dpsi1, dpsi2] — 8 dimensi
    Action: [v, w] continuous

    Reward mendorong:
    1. Jarak ke 2 neighbor → D_TARGET (sisi persegi)
    2. Sudut antara 2 neighbor → 90° (bentuk persegi, bukan rhombus)
    3. Heading alignment ke 2 neighbor → 0 (semua menghadap arah sama)
    4. Collision avoidance
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_id: int, seed: int = None):
        super().__init__()

        assert 0 <= robot_id < N_ROBOTS
        self.robot_id   = robot_id
        self.robot_name = ROBOT_NAMES[robot_id]
        self._rng = np.random.default_rng(
            seed if seed is not None else robot_id * 1000
        )

        obs_high = np.array([
            10.0, 10.0,
            10.0, 10.0,
             1.0,  2.0,
            np.pi, np.pi
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-V_MAX, -W_MAX], dtype=np.float32),
            high=np.array([ V_MAX,  W_MAX], dtype=np.float32),
            dtype=np.float32
        )

        _ensure_ros_init()

        self._cmd_pub = rospy.Publisher(
            f"/{self.robot_name}/cmd_vel", Twist, queue_size=1
        )

        if robot_id == 0:
            for name in ROBOT_NAMES:
                rospy.Subscriber(
                    f"/{name}/odom", Odometry,
                    _odom_callback, callback_args=name,
                    queue_size=1
                )
            rospy.loginfo("env_gazebo: odom subscribers registered")

        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
        self._set_state_srv = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        self._rate       = rospy.Rate(CONTROL_RATE)
        self._step_count = 0
        self._prev_v     = 0.0
        self._prev_w     = 0.0

        time.sleep(0.5)
        rospy.loginfo(f"GazeboFormationEnv [{self.robot_name}] ready.")

    # ── Gymnasium API ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_v     = 0.0
        self._prev_w     = 0.0

        if self.robot_id == 0:
            positions = _random_spawn_positions(
                N_ROBOTS, SPAWN_HALF, SPAWN_MIN_DIST, self._rng
            )
            for i, (rx, ry, ryaw) in enumerate(positions):
                self._teleport_robot(ROBOT_NAMES[i], rx, ry, ryaw)
            rospy.sleep(0.3)

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        v = float(np.clip(action[0], -V_MAX,  V_MAX))
        w = float(np.clip(action[1], -W_MAX,  W_MAX))

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        self._cmd_pub.publish(cmd)

        self._rate.sleep()
        self._step_count += 1
        self._prev_v = v
        self._prev_w = w

        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = False
        truncated  = self._step_count >= MAX_STEPS

        if truncated:
            self._stop_robot()

        info = {
            "step"      : self._step_count,
            "robot"     : self.robot_name,
            "collision" : self._check_collision(),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_robot()

    # ── Observasi ────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        State: [dx1, dy1, dx2, dy2, v_self, w_self, dpsi1, dpsi2]
        Semua dalam body frame robot ini.
        """
        me     = _shared_state[self.robot_name]
        v      = me["v"]
        w      = me["w"]
        others = self._get_sorted_neighbors()

        n1 = _shared_state[others[0][1]]
        n2 = _shared_state[others[1][1]]

        dx1, dy1 = self._to_body_frame(me, n1)
        dx2, dy2 = self._to_body_frame(me, n2)
        dpsi1    = self._wrap_angle(n1["yaw"] - me["yaw"])
        dpsi2    = self._wrap_angle(n2["yaw"] - me["yaw"])

        return np.array([
            dx1, dy1, dx2, dy2,
            me["v"], me["w"],
            dpsi1, dpsi2
        ], dtype=np.float32)

    # ── Reward ───────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Square formation reward.

        Komponen:
        - r_dist1, r_dist2 : jarak ke 2 neighbor → D_TARGET
        - r_angle          : sudut antara 2 neighbor → 90° (cos_angle → 0)
        - r_hdg            : heading alignment → dpsi1 + dpsi2 → 0
        - r_col            : collision penalty

        Catatan desain:
        r_angle diberi bobot ANGLE_W=5.0 untuk memaksa sudut 90°.
        r_far adalah penalty kuadrat untuk separasi > 2.5m —
        mencegah free rider problem (satu robot "kabur" dari kelompok).
        Tetap local observation: hanya butuh d1 dan d2.
        r_hdg diberi bobot 0.5*GAMMA agar tidak mendominasi geometri.
        """
        me     = _shared_state[self.robot_name]
        v      = me["v"]
        w      = me["w"]
        others = self._get_sorted_neighbors()

        n1 = _shared_state[others[0][1]]
        n2 = _shared_state[others[1][1]]
        d1 = others[0][0]
        d2 = others[1][0]

        dx1, dy1 = self._to_body_frame(me, n1)
        dx2, dy2 = self._to_body_frame(me, n2)
        dpsi1    = self._wrap_angle(n1["yaw"] - me["yaw"])
        dpsi2    = self._wrap_angle(n2["yaw"] - me["yaw"])

        # Spacing reward
        r_dist1 = -abs(d1 - D_TARGET)
        r_dist2 = -abs(d2 - D_TARGET)

        # Angle reward — sudut antara dua neighbor harus 90°
        # cos(90°) = 0, sehingga target: cos_angle → 0
        cos_angle = (dx1*dx2 + dy1*dy2) / (d1 * d2 + 1e-6)
        r_angle   = -abs(cos_angle)

        # Heading alignment — relatif ke 2 neighbor
        r_hdg = -abs(dpsi1) - abs(dpsi2)

        # Penalty kuadrat untuk separasi jauh — mencegah robot "kabur"
        # Aktif hanya jika jarak > 2.5m (> D_TARGET + margin)
        # Tetap local observation: hanya butuh d1 dan d2
        r_far = 0.0
        if d1 > 2.5:
            r_far -= 0.2 * (d1 - 2.5) ** 2
        if d2 > 2.5:
            r_far -= 0.2 * (d2 - 2.5) ** 2

        # Collision penalty
        r_col = -COLLISION_PEN if min(d1, d2) < D_SAFE else 0.0

        # Jerk penalty — kurangi maju-mundur dan oscillation
        r_smooth = -LAMBDA_SMOOTH * (abs(v - self._prev_v) + abs(w - self._prev_w))

        raw = (ALPHA * r_dist1 + ALPHA * r_dist2 +
               ANGLE_W * r_angle +
               0.5 * GAMMA * r_hdg +
               r_far +
               r_smooth +
               r_col)
        return float(raw / 10.0)

    # ── Termination ──────────────────────────────────────────────

    def _check_collision(self) -> bool:
        me = _shared_state[self.robot_name]
        for name in ROBOT_NAMES:
            if name == self.robot_name:
                continue
            other = _shared_state[name]
            if math.hypot(other["x"] - me["x"],
                          other["y"] - me["y"]) < D_SAFE:
                return True
        return False

    # ── Helper ───────────────────────────────────────────────────

    def _get_sorted_neighbors(self) -> list:
        me = _shared_state[self.robot_name]
        others = []
        for name in ROBOT_NAMES:
            if name == self.robot_name:
                continue
            d = math.hypot(
                _shared_state[name]["x"] - me["x"],
                _shared_state[name]["y"] - me["y"]
            )
            others.append((d, name))
        others.sort(key=lambda t: t[0])
        return others

    def _to_body_frame(self, me: dict, other: dict) -> tuple:
        dx_w = other["x"] - me["x"]
        dy_w = other["y"] - me["y"]
        yaw  = me["yaw"]
        dx_l =  dx_w * math.cos(yaw) + dy_w * math.sin(yaw)
        dy_l = -dx_w * math.sin(yaw) + dy_w * math.cos(yaw)
        return dx_l, dy_l

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _teleport_robot(self, name: str, x: float, y: float, yaw: float):
        state                    = ModelState()
        state.model_name         = name
        state.reference_frame    = "world"
        state.pose.position.x    = x
        state.pose.position.y    = y
        state.pose.position.z    = 0.1
        q = quaternion_from_euler(0, 0, yaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]
        state.twist              = Twist()
        try:
            self._set_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr(f"Teleport {name} gagal: {e}")

    def _stop_robot(self):
        self._cmd_pub.publish(Twist())


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Smoke test env_gazebo.py — Square Formation ===")
    print("Pastikan Gazebo + formasi_4_robot.launch sudah berjalan.")
    print()

    env = GazeboFormationEnv(robot_id=0)

    print("Test reset()...")
    obs, info = env.reset()
    print(f"  obs shape : {obs.shape}")
    print(f"  obs       : {np.round(obs, 3)}")

    print("\nTest step() x5...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: reward={reward:.4f}, "
              f"terminated={terminated}, truncated={truncated}")

    env.close()
    print("\nSmoke test selesai.")
