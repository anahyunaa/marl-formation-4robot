#!/usr/bin/env python3
from __future__ import annotations

"""
env_gazebo.py
=============
Gymnasium environment wrapper untuk IPPO training formasi 4 robot skid-steer.

Desain:
- Setiap robot diinstansiasi sebagai environment terpisah (IPPO / Independent Learners)
- Observasi lokal: k=2 nearest neighbor dalam body frame
- State: [dx1, dy1, dx2, dy2, v_self, w_self, dpsi1, dpsi2] — 8 dimensi
- Action: [v, w] continuous
- Reward: berbasis lateral alignment + longitudinal alignment + heading + collision

Penggunaan:
    env = GazeboFormationEnv(robot_id=0)   # robot index 0–3
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import rospy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion

import math
import time

# ─────────────────────────────────────────────
#  KONSTANTA GLOBAL
# ─────────────────────────────────────────────

ROBOT_NAMES     = ["r1", "r2", "r3", "r4"]
N_ROBOTS        = len(ROBOT_NAMES)

CONTROL_RATE    = 10        # Hz
MAX_STEPS       = 1000      # langkah per episode

# Action bounds
V_MAX           = 0.5       # m/s
W_MAX           = 1.0       # rad/s

# Formation parameters
D_TARGET        = 1.5       # meter — jarak target antar robot (side-by-side)
D_SAFE          = 0.75      # meter — jarak minimum sebelum collision penalty

# Reward weights  (tunable)
ALPHA           = 1.0       # bobot lateral error   ||dy| - d*|
BETA            = 1.0       # bobot longitudinal error |dx|
GAMMA           = 0.5       # bobot heading error   |dpsi|
COLLISION_PEN   = 50.0      # penalty tabrakan (flat)
                              # HARUS jauh lebih besar dari max reward harian (~6/step)
                              # agar robot tidak exploit early termination

# Spawn area (random reset)
SPAWN_HALF      = 2.0       # meter — robot di-spawn dalam kotak ±2m dari pusat arena
SPAWN_MIN_DIST  = 1.0       # meter — jarak minimum antar robot saat spawn

# Curriculum Learning flag
# Phase 1 (validasi reward): RANDOM_YAW = False → semua robot spawn yaw=0.0
# Phase 2 (training final) : RANDOM_YAW = True  → yaw acak
# Ganti nilai ini setelah reward terbukti konvergen di Phase 1
RANDOM_YAW      = False     # ← mulai dari False dulu


# ─────────────────────────────────────────────
#  SHARED STATE — dibaca semua instance env
# ─────────────────────────────────────────────
# Dict ini di-update oleh odom callback semua robot.
# Karena ROS berjalan dalam satu proses Python, shared dict ini aman.

_shared_state = {
    name: {"x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0, "w": 0.0}
    for name in ROBOT_NAMES
}
_ros_initialized = False


def _ensure_ros_init():
    """Inisialisasi ROS node sekali saja untuk seluruh proses."""
    global _ros_initialized
    if not _ros_initialized:
        rospy.init_node("marl_formation_env", anonymous=False, disable_signals=True)
        _ros_initialized = True


def _odom_callback(msg: Odometry, robot_name: str):
    """Update shared state dari odometry — dipanggil oleh ROS subscriber."""
    x   = msg.pose.pose.position.x
    y   = msg.pose.pose.position.y
    q   = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    v   = msg.twist.twist.linear.x
    w   = msg.twist.twist.angular.z

    _shared_state[robot_name]["x"]   = x
    _shared_state[robot_name]["y"]   = y
    _shared_state[robot_name]["yaw"] = yaw
    _shared_state[robot_name]["v"]   = v
    _shared_state[robot_name]["w"]   = w


def _random_spawn_positions(n: int, half: float, min_dist: float,
                             rng: np.random.Generator) -> list[tuple]:
    """
    Generate posisi spawn acak untuk n robot dalam kotak ±half meter.
    Rejection sampling untuk memastikan jarak minimum antar robot terpenuhi.
    Maksimum 1000 percobaan sebelum fallback ke grid.
    """
    positions = []
    max_attempts = 1000

    for _ in range(max_attempts):
        candidate = (
            rng.uniform(-half, half),
            rng.uniform(-half, half),
            rng.uniform(-math.pi, math.pi) if RANDOM_YAW else 0.0,   # yaw: acak atau 0 (curriculum)
        )
        too_close = any(
            math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < min_dist
            for p in positions
        )
        if not too_close:
            positions.append(candidate)
        if len(positions) == n:
            return positions

    # Fallback: grid 2×2 dengan sedikit noise — tidak pernah seharusnya tercapai
    rospy.logwarn("spawn: fallback ke grid (rejection sampling gagal)")
    grid = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    return [(gx + rng.uniform(-0.1, 0.1),
             gy + rng.uniform(-0.1, 0.1),
             rng.uniform(-math.pi, math.pi)) for gx, gy in grid[:n]]


# ─────────────────────────────────────────────
#  ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class GazeboFormationEnv(gym.Env):
    """
    Gymnasium environment untuk satu robot dalam sistem IPPO 4-robot.

    Setiap robot diinstansiasi sebagai env terpisah:
        env0 = GazeboFormationEnv(robot_id=0)  # kontrol r1
        env1 = GazeboFormationEnv(robot_id=1)  # kontrol r2
        ...

    Semua env berbagi _shared_state yang di-update oleh subscriber ROS.
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_id: int, seed: int = None):
        super().__init__()

        assert 0 <= robot_id < N_ROBOTS, f"robot_id harus 0–{N_ROBOTS-1}"
        self.robot_id   = robot_id
        self.robot_name = ROBOT_NAMES[robot_id]

        # RNG pribadi per env — seed berbeda per robot untuk eksplorasi beragam
        self._rng = np.random.default_rng(seed if seed is not None
                                          else robot_id * 1000)

        # ── Spaces ──────────────────────────────────────────────
        # State: [dx1, dy1, dx2, dy2, v_self, w_self, dpsi1, dpsi2]
        # Batas obs dibuat longgar — normalisasi dilakukan di _get_obs()
        obs_high = np.array([
            10.0, 10.0,   # dx1, dy1
            10.0, 10.0,   # dx2, dy2
             1.0,  2.0,   # v_self, w_self
            np.pi, np.pi  # dpsi1, dpsi2
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Action: [v, w]
        self.action_space = spaces.Box(
            low=np.array([-V_MAX, -W_MAX], dtype=np.float32),
            high=np.array([ V_MAX,  W_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # ── ROS setup ───────────────────────────────────────────
        _ensure_ros_init()

        # Publisher cmd_vel untuk robot ini
        self._cmd_pub = rospy.Publisher(
            f"/{self.robot_name}/cmd_vel", Twist, queue_size=1
        )

        # Subscriber odom untuk SEMUA robot (update shared state)
        # Hanya robot_id=0 yang register subscriber — robot lain menumpang shared state
        # Ini menghindari 4×4 = 16 subscriber yang redundan
        if robot_id == 0:
            for name in ROBOT_NAMES:
                rospy.Subscriber(
                    f"/{name}/odom", Odometry,
                    _odom_callback, callback_args=name,
                    queue_size=1
                )
            rospy.loginfo("env_gazebo: subscriber odom terdaftar untuk semua robot")

        # Service set_model_state untuk teleport (reset episode)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
        self._set_state_srv = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        # Rate control
        self._rate = rospy.Rate(CONTROL_RATE)

        # Step counter
        self._step_count = 0

        # Beri waktu subscriber connect
        time.sleep(0.5)
        rospy.loginfo(f"GazeboFormationEnv [{self.robot_name}] siap.")

    # ── Gymnasium API ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset episode: teleport semua robot ke posisi acak, return observasi awal.
        Hanya robot_id=0 yang melakukan teleport (menghindari race condition).
        """
        super().reset(seed=seed)
        self._step_count = 0

        # Hanya env robot 0 yang teleport semua robot
        # env lain menunggu shared_state ter-update via odom
        if self.robot_id == 0:
            positions = _random_spawn_positions(
                N_ROBOTS, SPAWN_HALF, SPAWN_MIN_DIST, self._rng
            )
            for i, (rx, ry, ryaw) in enumerate(positions):
                self._teleport_robot(ROBOT_NAMES[i], rx, ry, ryaw)

            # Beri waktu Gazebo memproses teleport dan odom ter-update
            rospy.sleep(0.3)

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Eksekusi satu langkah kontrol:
        1. Publish action (v, w) ke cmd_vel
        2. Tunggu satu cycle (1/CONTROL_RATE detik)
        3. Baca observasi baru
        4. Hitung reward
        5. Cek terminasi
        """
        # Clamp action ke bounds (defensive)
        v = float(np.clip(action[0], -V_MAX,  V_MAX))
        w = float(np.clip(action[1], -W_MAX,  W_MAX))

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        self._cmd_pub.publish(cmd)

        self._rate.sleep()
        self._step_count += 1

        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._check_collision()          # tabrakan → episode selesai
        truncated  = self._step_count >= MAX_STEPS    # timeout

        # Stop robot saat episode berakhir
        if terminated or truncated:
            self._stop_robot()

        info = {
            "step":          self._step_count,
            "robot":         self.robot_name,
            "collision":     terminated,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_robot()

    # ── Observasi ────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Bangun state vector 8-dimensi dalam body frame robot ini.

        State: [dx1, dy1, dx2, dy2, v_self, w_self, dpsi1, dpsi2]
        - dx, dy: posisi relatif tetangga dalam body frame robot (lokal)
        - v_self, w_self: kecepatan linier dan angular robot ini
        - dpsi: selisih heading robot ini dengan tetangga (dalam [-π, π])
        """
        me = _shared_state[self.robot_name]

        # Hitung jarak ke semua robot lain
        others = []
        for name in ROBOT_NAMES:
            if name == self.robot_name:
                continue
            other = _shared_state[name]
            dist  = math.hypot(other["x"] - me["x"], other["y"] - me["y"])
            others.append((dist, name))

        # Sort by distance → ambil 2 terdekat
        others.sort(key=lambda t: t[0])
        n1_name = others[0][1]
        n2_name = others[1][1]

        # Transformasi ke body frame
        dx1, dy1 = self._to_body_frame(me, _shared_state[n1_name])
        dx2, dy2 = self._to_body_frame(me, _shared_state[n2_name])

        # Heading difference — wrap ke [-π, π]
        dpsi1 = self._wrap_angle(
            _shared_state[n1_name]["yaw"] - me["yaw"]
        )
        dpsi2 = self._wrap_angle(
            _shared_state[n2_name]["yaw"] - me["yaw"]
        )

        obs = np.array([
            dx1, dy1,
            dx2, dy2,
            me["v"], me["w"],
            dpsi1, dpsi2
        ], dtype=np.float32)

        return obs

    # ── Reward ───────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Reward berbasis nearest neighbor terdekat (neighbor 1).

        Komponen:
        - r_lat  : lateral error — |dy1| harus mendekati D_TARGET
        - r_lon  : longitudinal error — |dx1| harus mendekati 0
        - r_hdg  : heading alignment — |dpsi1| harus mendekati 0
        - r_col  : collision penalty (flat) jika jarak < D_SAFE

        Catatan desain:
        Hanya neighbor 1 (terdekat) yang dipakai untuk reward spacing.
        Neighbor 2 masuk state sebagai konteks tambahan tetapi tidak
        secara eksplisit dipaksa ke jarak tertentu — menghindari
        paradoks spacing untuk robot ujung formasi.
        """
        me = _shared_state[self.robot_name]

        # Nearest neighbor
        others = []
        for name in ROBOT_NAMES:
            if name == self.robot_name:
                continue
            other = _shared_state[name]
            dist  = math.hypot(other["x"] - me["x"], other["y"] - me["y"])
            others.append((dist, name))
        others.sort(key=lambda t: t[0])
        n1 = _shared_state[others[0][1]]
        d1 = others[0][0]

        dx1, dy1 = self._to_body_frame(me, n1)
        dpsi1    = self._wrap_angle(n1["yaw"] - me["yaw"])

        # Komponen reward
        r_lat = -abs(abs(dy1) - D_TARGET)   # lateral: |dy1| → D_TARGET
        r_lon = -abs(dx1)                   # longitudinal: dx1 → 0
        r_hdg = -abs(dpsi1)                 # heading: dpsi1 → 0

        # Collision penalty
        r_col = -COLLISION_PEN if d1 < D_SAFE else 0.0

        reward = ALPHA * r_lat + BETA * r_lon + GAMMA * r_hdg + r_col
        return float(reward)

    def _check_collision(self) -> bool:
        """
        Cek apakah robot ini terlalu dekat dengan robot manapun.
        Jarak < D_SAFE → terminated = True.
        """
        me = _shared_state[self.robot_name]
        for name in ROBOT_NAMES:
            if name == self.robot_name:
                continue
            other = _shared_state[name]
            dist  = math.hypot(other["x"] - me["x"], other["y"] - me["y"])
            if dist < D_SAFE:
                return True
        return False

    # ── Helper ───────────────────────────────────────────────────

    def _to_body_frame(self, me: dict, other: dict) -> tuple[float, float]:
        """
        Transformasi posisi other ke body frame robot me.

        Body frame ROS (REP-103):
        - X: arah depan robot
        - Y: arah kiri robot

        Returns: (dx_local, dy_local)
        """
        dx_world = other["x"] - me["x"]
        dy_world = other["y"] - me["y"]
        yaw      = me["yaw"]

        dx_local =  dx_world * math.cos(yaw) + dy_world * math.sin(yaw)
        dy_local = -dx_world * math.sin(yaw) + dy_world * math.cos(yaw)

        return dx_local, dy_local

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap sudut ke range [-π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _teleport_robot(self, name: str, x: float, y: float, yaw: float):
        """Teleport robot ke posisi (x, y, yaw) via Gazebo set_model_state."""
        from geometry_msgs.msg import Pose, Twist as TwistMsg
        from tf.transformations import quaternion_from_euler

        state      = ModelState()
        state.model_name        = name
        state.reference_frame   = "world"

        state.pose.position.x   = x
        state.pose.position.y   = y
        state.pose.position.z   = 0.1

        q = quaternion_from_euler(0, 0, yaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]

        # Reset velocity
        state.twist = TwistMsg()

        try:
            self._set_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr(f"Teleport {name} gagal: {e}")

    def _stop_robot(self):
        """Kirim perintah berhenti ke robot ini."""
        self._cmd_pub.publish(Twist())


if __name__ == "__main__":
    print("=== Smoke test env_gazebo.py ===")
    print("Pastikan Gazebo + formasi_4_robot.launch sudah berjalan.")
    print()

    env = GazeboFormationEnv(robot_id=0)

    print("Test reset()...")
    obs, info = env.reset()
    print(f"  obs shape : {obs.shape}")
    print(f"  obs       : {obs}")

    print("\nTest step() x5 dengan aksi random...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: reward={reward:.4f}, "
              f"terminated={terminated}, truncated={truncated}")

    env.close()
    print("\nSmoke test selesai.")
