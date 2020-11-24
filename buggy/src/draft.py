import numpy as np
import quaternion as q
import math as m

def q2e(quat):
    w, x, y, z = quat
    pitch = -m.asin(2.0 * (x * z - w * y))
    roll = (
            m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    )
    yaw = (
            m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    )

    return roll, pitch, yaw


def e2q(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qw, qx, qy, qz]

for i in range(1000):
    rnd_euler = np.random.randn(3)
    rnd_quat = list(np.random.rand(4))
    rnd_quat_normed = rnd_quat / np.linalg.norm(rnd_quat)
    euler_my_res = q2e(rnd_quat_normed)
    euler_lib_res = q.as_euler_angles(q.quaternion(*rnd_quat_normed))
    quat_my_res = e2q(*q2e(rnd_quat_normed))
    quat_lib_res = q.from_euler_angles(q.as_euler_angles(q.quaternion(*rnd_quat_normed))).components
    #euler_my_res = q2e(e2q(*rnd_euler))
    #euler_lib_res = q.as_euler_angles(q.from_euler_angles(*rnd_euler))
    if not np.allclose(rnd_quat_normed, quat_my_res, rtol=0.001):
         print(rnd_quat_normed, quat_my_res)
    #if not np.allclose(euler_my_res, euler_lib_res, rtol=0.001):
    #    print(euler_my_res, euler_lib_res)
