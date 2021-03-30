import numpy as np

def rads_to_norm(joints, low, diff):
    sjoints = ((np.array(joints) - low) / diff) * 2 - 1
    return sjoints

def norm_to_rads(joints, low, diff):
    return (np.array(joints) * 0.5 + 0.5) * diff + low

def rads_to_servo(joints):
    return (((np.array(joints) / 5.23599) + 0.5) * 1023).astype(np.uint16)

def servo_to_rads(joints):
    return ((joints / 1023) - 0.5) * 5.23599

def print_sometimes(msg, prob=0.01):
    if np.random.rand() < prob:
        print(msg)

def my_ikt(target_positions, rotation_overlay=None):
    # raise NotImplementedError
    rotation_angles = np.array([np.pi / 4, np.pi / 4, 0, 0, -np.pi / 4, -np.pi / 4])
    if rotation_overlay is not None:
        rotation_angles += rotation_overlay
    joint_angles = []
    for i, tp in enumerate(target_positions):
        tp_rotated = rotate_eef_pos(tp, rotation_angles[i], tp[1])
        joint_angles.extend(single_leg_ikt(tp_rotated))
    return joint_angles

def my_ikt_robust(target_positions, rotation_overlay=None):
    # raise NotImplementedError
    def find_nearest_valid_point(xyz_query, rot_angle=0):
        sol = single_leg_ikt(xyz_query)
        if not np.isnan(sol).any(): return sol

        cur_valid_sol = None
        cur_xyz_query = xyz_query
        cur_delta = 0.03
        n_iters = 10

        if xyz_query[2] > -0.1:
            search_dir = 1
        else:
            search_dir = -1

        cur_xyz_query[0] = cur_xyz_query[0] - cur_delta * search_dir * np.sin(rot_angle)
        cur_xyz_query[1] = cur_xyz_query[1] + cur_delta * search_dir * np.cos(rot_angle)
        for _ in range(n_iters):
            sol = single_leg_ikt(cur_xyz_query)
            if not np.isnan(sol).any():  # If solution is good
                cur_valid_sol = sol
                cur_delta /= 2
                cur_xyz_query[0] = cur_xyz_query[0] + cur_delta * search_dir * np.sin(rot_angle)
                cur_xyz_query[1] = cur_xyz_query[1] - cur_delta * search_dir * np.cos(rot_angle)
            else:
                if cur_valid_sol is not None:
                    cur_delta /= 2
                cur_xyz_query[0] = cur_xyz_query[0] - cur_delta * search_dir * np.sin(rot_angle)
                cur_xyz_query[1] = cur_xyz_query[1] + cur_delta * search_dir * np.cos(rot_angle)

        assert cur_valid_sol is not None and not np.isnan(cur_valid_sol).any()
        return cur_valid_sol

    rotation_angles = np.array([np.pi / 4, np.pi / 4, 0, 0, -np.pi / 4, -np.pi / 4])
    if rotation_overlay is not None:
        rotation_angles += rotation_overlay
    joint_angles = []
    for i, tp in enumerate(target_positions):
        tp_rotated = rotate_eef_pos(tp, rotation_angles[i], tp[1])
        joint_angles.extend(find_nearest_valid_point(tp_rotated, rotation_angles[i]))
    return joint_angles

def rotate_eef_pos(eef_xyz, angle, y_offset):
    return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

def single_leg_ikt(eef_xyz):
    x, y, z = eef_xyz

    q1 = 0.2137
    q2 = 0.785

    C = 0.052
    F = 0.0675
    T = 0.132

    psi = np.arctan(x / y)
    Cx = C * np.sin(psi)
    Cy = C * np.cos(psi)
    R = np.sqrt((x - Cx) ** 2 + (y - Cy) ** 2 + (z) ** 2)
    alpha = np.arcsin(-z / R)

    a = np.arccos((F ** 2 + R ** 2 - T ** 2) / (2 * F * R))
    b = np.arccos((F ** 2 + T ** 2 - R ** 2) / (2 * F * T))

    # if np.isnan(a) or np.isnan(b):
    #    print(a,b)

    assert 0 < a < np.pi or np.isnan(a)
    assert 0 < b < np.pi or np.isnan(b)

    th1 = alpha - q1 - a
    th2 = np.pi - q2 - b

    assert th2 + q2 > 0 or np.isnan(th2)

    return -psi, th1, th2

def single_leg_dkt(angles):
    psi, th1, th2 = angles

    q1 = 0.2137
    q2 = 0.785

    C = 0.052
    F = 0.0675
    T = 0.132

    Ey_flat = (C + F * np.cos(q1 + th1) + T * np.cos(q1 + th1 + q2 + th2))

    Ez = - F * np.sin(q1 + th1) - T * np.sin(q1 + th1 + q2 + th2)
    Ey = Ey_flat * np.cos(psi)
    Ex = Ey_flat * np.sin(-psi)

    return (Ex, Ey, Ez)