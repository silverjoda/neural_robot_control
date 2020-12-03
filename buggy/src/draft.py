def calculate_stabilization_action(self, orientation, angular_velocities, targets):
    roll, pitch, _ = p.getEulerFromQuaternion(orientation)
    roll_vel, pitch_vel, yaw_vel = angular_velocities
    t_throttle, t_roll, t_pitch, t_yaw_vel = targets

    # Increase t_yaw_vel because it's slow as shit
    t_yaw_vel *= 5

    # print(f"Throttle_target: {t_throttle}, Roll_target: {t_roll}, Pitch_target: {t_pitch}, Yaw_vel_target: {t_yaw_vel}")
    # print(f"Roll: {roll}, Pitch: {pitch}, Yaw_vel: {yaw_vel}")

    # Target errors
    e_roll = t_roll - roll
    e_pitch = t_pitch - pitch
    e_yaw = t_yaw_vel - yaw_vel

    # Desired correction action
    roll_act = e_roll * self.p_roll + (e_roll - self.e_roll_prev) * self.d_roll
    pitch_act = e_pitch * self.p_pitch + (e_pitch - self.e_pitch_prev) * self.d_pitch
    yaw_act = e_yaw * self.p_yaw + (e_yaw - self.e_yaw_prev) * self.d_yaw

    self.e_roll_prev = e_roll
    self.e_pitch_prev = e_pitch
    self.e_yaw_prev = e_yaw

    m_1_act_total = + roll_act - pitch_act + yaw_act
    m_2_act_total = - roll_act - pitch_act - yaw_act
    m_3_act_total = + roll_act + pitch_act - yaw_act
    m_4_act_total = - roll_act + pitch_act + yaw_act

    # Translate desired correction actions to servo commands
    m_1 = np.clip(t_throttle + m_1_act_total, 0, 1)
    m_2 = np.clip(t_throttle + m_2_act_total, 0, 1)
    m_3 = np.clip(t_throttle + m_3_act_total, 0, 1)
    m_4 = np.clip(t_throttle + m_4_act_total, 0, 1)

    # print([m_1, m_2, m_3, m_4])

    if np.max([m_1, m_2, m_3, m_4]) > 1.1:
        print("Warning: motor commands exceed 1.0. This signifies an error in the system", m_1, m_2, m_3, m_4,
              t_throttle)

    return m_1, m_2, m_3, m_4