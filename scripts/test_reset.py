# Di terminal Python, atau buat file test_reset.py sementara
import rospy
import sys
sys.path.insert(0, '/home/anahyuna/marl_ws/src/marl_formation/scripts')

from env_gazebo import GazeboFormationEnv, _shared_state, ROBOT_NAMES

env = GazeboFormationEnv(robot_id=0)

for episode in range(3):
    obs, info = env.reset()
    import rospy
    rospy.sleep(0.5)  # beri waktu odom update

    print(f"\n=== Episode {episode+1} — posisi setelah reset ===")
    for name in ROBOT_NAMES:
        s = _shared_state[name]
        print(f"  {name}: x={s['x']:.3f}, y={s['y']:.3f}, yaw={s['yaw']:.3f}")

env.close()
