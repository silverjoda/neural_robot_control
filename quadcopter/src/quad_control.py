import Adafruit_PCA9685
import sys
import threading
import copy
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time
import torch as T
import torch.nn as nn
import torch.functional as F
import numpy as np
import logging
import smbus
import pyrealsense2 as rs
logging.basicConfig(level=logging.INFO)
import pygame
import os
import yaml
import math as m
import quaternion
import random
from stable_baselines import A2C


# root = logging.getLogger()
# root.setLevel(logging.INFO)
#
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root.addHandler(handler)
#  m1(cw)   m2(ccw)
#      -     -
#        - -
#        - -
#      -     -
#  m3(ccw)  m4(cw)
#
# Target inputs are in radians
# Motor inputs to PWM driver are in [0,1], later scaled to pulsewidth 1ms-2ms
# Gyro scale is maximal (+-2000 deg/s)
# Acc scale is ??


class JoyController():
    def __init__(self, config):
        self.config = config
        logging.info("Initializing joystick controller")
        pygame.init()
        if self.config["target_vel_source"] == "joystick":
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        else:
            logging.info("No joystick found")
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        throttle, t_roll, t_pitch, t_yaw = \
                [self.joystick.get_axis(self.config["joystick_mapping"][i]) for i in range(4)]
        button_x = self.joystick.get_button(0)
        pygame.event.clear()

        # button_x only when upon press
        # if self.button_x_state == 0 and button_x == 1:
        #     self.button_x_state = 1
        #     button_x = 1
        # elif self.button_x_state == 1 and button_x == 0:
        #     self.button_x_state = 0
        #     button_x = 0
        # elif self.button_x_state == 1 and button_x == 1:
        #     self.button_x_state = 1
        #     button_x = 0
        # else:
        #     self.button_x_state = 0
        #     button_x = 0

        return -throttle, t_roll, -t_pitch, -t_yaw, button_x


class AHRS:
    DEVICE_ADDR = 0x68  # MPU6050 device address
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    INT_ENABLE = 0x38
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    GYRO_XOUT_H = 0x43
    GYRO_YOUT_H = 0x45
    GYRO_ZOUT_H = 0x47
    GYRO_SCALER = (2 * np.pi / 360. ) * (2000. / (2 ** 15)) # +- 2000 dps across a signed 16 bit value 
    ACC_SENSITIVITY = 16384.0

    def __init__(self):
        print("Initializing the MPU6050. ")

        self.bus = smbus.SMBus(1)

        # write to sample rate register
        self.bus.write_byte_data(AHRS.DEVICE_ADDR, AHRS.SMPLRT_DIV, 7)

        # Write to power management register
        self.bus.write_byte_data(AHRS.DEVICE_ADDR, AHRS.PWR_MGMT_1, 1)

        # Write to Configuration register
        self.bus.write_byte_data(AHRS.DEVICE_ADDR, AHRS.CONFIG, 0)

        # Write to Gyro configuration register
        self.bus.write_byte_data(AHRS.DEVICE_ADDR, AHRS.GYRO_CONFIG, 24)

        # Write to interrupt enable register
        self.bus.write_byte_data(AHRS.DEVICE_ADDR, AHRS.INT_ENABLE, 1)

        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0

        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.quat = [1, 0, 0, 0]

        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0

        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0

        self.gyro_integration_coeff = 0.95
        self.acc_integration_coeff = 1.0 - self.gyro_integration_coeff

        self.timestamp = time.time()
        
        print("Finished initializing the MPU6050. ")

    def _read_raw_data(self, addr):
        # Accelero and Gyro value are 16-bit
        high = self.bus.read_byte_data(AHRS.DEVICE_ADDR, addr)
        low = self.bus.read_byte_data(AHRS.DEVICE_ADDR, addr + 1)

        # concatenate higher and lower value
        value = ((high << 8) | low)

        # to get signed value from mpu6050
        if (value > 32768):
            value = value - 65536
        return value

    def update(self):
        # Read Accelerometer raw value
        self.acc_x = self._read_raw_data(
            AHRS.ACCEL_XOUT_H) / AHRS.ACC_SENSITIVITY
        self.acc_y = self._read_raw_data(
            AHRS.ACCEL_YOUT_H) / AHRS.ACC_SENSITIVITY
        self.acc_z = self._read_raw_data(
            AHRS.ACCEL_ZOUT_H) / AHRS.ACC_SENSITIVITY

        # Read Gyroscope raw value
        self.gyro_x = self._read_raw_data(
            AHRS.GYRO_XOUT_H) * AHRS.GYRO_SCALER
        self.gyro_y = self._read_raw_data(
            AHRS.GYRO_YOUT_H) * AHRS.GYRO_SCALER
        self.gyro_z = self._read_raw_data(
            AHRS.GYRO_ZOUT_H) * AHRS.GYRO_SCALER

        # Calculate roll, pitch and yaw and corresponding quaternion
        t = time.time()
        dt = t - self.timestamp

        acc_x_dir = np.arctan2(self.acc_x, np.sqrt(
            self.acc_y ** 2 + self.acc_z ** 2))
        acc_y_dir = np.arctan2(self.acc_y, np.sqrt(
            self.acc_x ** 2 + self.acc_z ** 2))
        acc_z_dir = np.arctan2(self.acc_z, np.sqrt(
            self.acc_x ** 2 + self.acc_y ** 2))

        acc_z_dir_gravity_corrected = acc_z_dir + 9.81

        self.roll = self.gyro_integration_coeff * \
            (self.roll + self.gyro_x * dt) + \
            self.acc_integration_coeff * acc_y_dir
        self.pitch = self.gyro_integration_coeff * \
            (self.pitch + self.gyro_y * dt) - \
            self.acc_integration_coeff * acc_x_dir
        self.yaw = self.yaw + self.gyro_z * dt
        quat = quaternion.from_euler_angles(self.roll, self.pitch, self.yaw)
        self.quat = [quat.w, quat.x, quat.y, quat.z]

        self.timestamp = t

        self.vel_x = self.vel_x + dt * acc_x_dir
        self.vel_y = self.vel_y + dt * acc_y_dir
        self.vel_z = self.vel_z + dt * acc_z_dir_gravity_corrected

        self.pos_x = self.pos_x + dt * self.vel_x
        self.pos_y = self.pos_y + dt * self.vel_y
        self.pos_z = self.pos_z + dt * self.vel_z

        vel_rob = [self.vel_x, self.vel_y, self.vel_z]
        pos_rob = [self.pos_x, self.pos_y, self.pos_z]
        angular_vel_rob = [self.gyro_x, self.gyro_y, self.gyro_z]
        euler_rob = [self.roll, self.pitch, self.yaw]

        #position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, timestamp
        return pos_rob, vel_rob, self.quat, angular_vel_rob, euler_rob, self.timestamp


class AHRS_RS:
    def __init__(self):
        print("Initializing the rs_t265. ")

        self.rs_to_world_mat = np.array([[0, 0, 1],
                                         [1, 0, 0],
                                         [0, 1, 0]])

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        self.pipe.start(self.cfg, callback=self.rs_cb)
        self.timestamp = time.time()
        self.rs_lock = threading.Lock()

        self.rs_frame = None
        print("Finished initializing the rs_t265. ")

    def rs_cb(self, data_frame):
        with self.rs_lock:
            self.rs_frame = data_frame

    def _quat_to_euler(self, w, x, y, z):
        pitch =  -m.asin(2.0 * (x*z - w*y))
        roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return (roll, pitch, yaw)

    def update(self):
        self.timestamp = time.time()

        if self.rs_frame is not None:
            with self.rs_lock:
                data = self.rs_frame.as_pose_frame().get_pose_data()

            position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
            vel_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            position_rob = self.rs_to_world_mat @ position_rs
            vel_rob = self.rs_to_world_mat @ vel_rs

            # Axes are permuted according how the RS axes are oriented wrt world axes
            rotation_rob = (data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y)
            angular_vel_rob = (data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y)
            euler_rob = self._quat_to_euler(*rotation_rob)
        else:
            position_rob = [0., 0., 0.]
            vel_rob = [0., 0., 0.]
            rotation_rob = [0., 0., 0., 0.]
            angular_vel_rob = [0., 0., 0.]
            euler_rob = [0., 0., 0.]

        return position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, self.timestamp


class PWMDriver:
    def __init__(self, motors_on):
        self.motors_on = motors_on

        self.pwm_freq = 200 
        self.servo_ids = [0, 1, 2, 3]

        print("Initializing the PWMdriver. ")
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(self.pwm_freq)

        self.arm_escs()

        print("Finished initializing the PWMdriver. ")

    def write_servos(self, vals):
        '''

        :param vals: Throttle commands [0,1] corresponding to min and max values
        :return: None
        '''
        
        if not self.motors_on:
            vals = [0,0,0,0]

        for id in self.servo_ids:
            pulse_length = ((np.clip(vals[id], 0, 1) + 1) * 1000) / ((1000000. / self.pwm_freq) / 4096.)
            self.pwm.set_pwm(id, 0, int(pulse_length))

    def arm_escs(self):
        print("Setting escs to lowest value. ")
        self.write_servos([0, 0, 0, 0])

        time.sleep(0.3)


class Controller:
    def __init__(self):
        with open('configs/default.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.motors_on = self.config["motors_on"]

        print("Initializing the Controller, motors_on: {}".format(self.motors_on))

        self.config["obs_dim"], self.config["act_dim"] = 13, 4

        if self.config["ahrs_source"] == "rs":
            self.AHRS = AHRS_RS()
        else:
            self.AHRS = AHRS()

        self.PWMDriver = PWMDriver(self.motors_on)
        self.JOYStick = JoyController(self.config)
        self.policy = self.load_policy(self.config)
        self.setup_stabilization_control()

        self.obs_queue = [np.zeros(self.config["obs_dim"], dtype=np.float32) for _ in range(
            np.maximum(1, self.config["obs_input"]))]
        self.act_queue = [np.zeros(self.config["act_dim"], dtype=np.float32) for _ in range(
            np.maximum(1, self.config["act_input"]))]
        self.rew_queue = [np.zeros(1, dtype=np.float32) for _ in range(
            np.maximum(1, self.config["rew_input"]))]

        print("Finished initializing the Controller")

    def setup_stabilization_control(self):
        self.p_roll = 0.2
        self.p_pitch = 0.2
        self.p_yaw = 0.1

        self.d_roll = 2.0
        self.d_pitch = 2.0
        self.d_yaw = 0.01

        self.i_roll = -0.03
        self.i_pitch = -0.03

        self.e_roll_prev = 0
        self.e_pitch_prev = 0
        self.e_yaw_prev = 0

        self.e_roll_accum = 0
        self.e_pitch_accum = 0

    def load_policy(self, config):
        logging.info("Loading policy from: \"{}\" ".format(config["policy_path"]))
        policy = A2C.load("agents/{}".format(config["policy_path"]))
        return policy

    def get_policy_action(self, obs):
        (m1, m2, m3, m4), _ = self.policy.predict(obs)
        return [m1, m2, m3, m4]

    def calculate_stabilization_action(self, orientation_euler, angular_velocities, targets):
        roll, pitch, _ = orientation_euler
        roll_vel, pitch_vel, yaw_vel = angular_velocities
        t_throttle, t_roll, t_pitch, t_yaw_vel = targets
        # print(orientation_euler, targets)

        # print(f"Throttle_target: {t_throttle}, Roll_target: {t_roll}, Pitch_target: {t_pitch}, Yaw_vel_target: {t_yaw_vel}")
        # print(f"Roll: {roll}, Pitch: {pitch}, Yaw_vel: {yaw_vel}")

        # Target errors
        e_roll = t_roll - roll
        e_pitch = t_pitch - pitch
        e_yaw = t_yaw_vel - yaw_vel

        decay_fac = 0.7
        self.e_roll_accum = self.e_roll_accum * decay_fac + e_roll
        self.e_pitch_accum = self.e_pitch_accum * decay_fac + e_pitch

        # Desired correction action
        roll_act = e_roll * self.p_roll + (e_roll - self.e_roll_prev) * self.d_roll + self.e_roll_accum * self.i_roll
        pitch_act = e_pitch * self.p_pitch + (e_pitch - self.e_pitch_prev) * self.d_pitch + self.e_pitch_accum * self.i_pitch
        yaw_act = e_yaw * self.p_yaw + (e_yaw - self.e_yaw_prev) * self.d_yaw

        self.e_roll_prev = e_roll
        self.e_pitch_prev = e_pitch
        self.e_yaw_prev = e_yaw

        #print(e_roll, roll_act)

        m_1_act_total = + roll_act - pitch_act + yaw_act
        m_2_act_total = - roll_act - pitch_act - yaw_act
        m_3_act_total = + roll_act + pitch_act - yaw_act
        m_4_act_total = - roll_act + pitch_act + yaw_act

        # Translate desired correction actions to servo commands
        m_1 = np.clip(t_throttle + m_1_act_total, 0, 1) * 2 - 1
        m_2 = np.clip(t_throttle + m_2_act_total, 0, 1) * 2 - 1
        m_3 = np.clip(t_throttle + m_3_act_total, 0, 1) * 2 - 1
        m_4 = np.clip(t_throttle + m_4_act_total, 0, 1) * 2 - 1

        return [m_1, m_2, m_3, m_4]

    def step(self, ctrl_raw):
        self.act_queue.append(ctrl_raw)
        self.act_queue.pop(0)
        act_raw_unqueued = self.act_queue

        act_normed = np.clip(ctrl_raw, -1, 1) * 0.5 + 0.5

        # Write control to servos
        t_pwm_1 = time.time()
        self.PWMDriver.write_servos(act_normed)
        t_pwm_2 = time.time()
        if self.config["time_prints"]: print(f"PWM write pass took: {t_pwm_2 - t_pwm_1}")

        # Read target control inputs
        t_js_1 = time.time()
        throttle, t_roll, t_pitch, t_yaw, autonomous_control = self.JOYStick.get_joystick_input()
        t_js_2 = time.time()
        if self.config["time_prints"]: print(f"JS read pass took: {t_js_2 - t_js_1}")
        velocity_targets = throttle, -t_roll, t_pitch, t_yaw
        pid_targets = throttle, t_roll, t_pitch, t_yaw

        # Update sensor data
        t_ahrs_1 = time.time()
        position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, timestamp = self.AHRS.update()
        t_ahrs_2 = time.time()
        if self.config["time_prints"]: print(f"AHRS read $ process pass took: {t_ahrs_2 - t_ahrs_1}")
        roll, pitch, yaw = euler_rob
        pos_delta = np.array(position_rob) + np.array(self.config["starting_pos"]) - np.array(self.config["target_pos"])

        # Calculate reward
        p_position = np.clip(np.mean(np.square(pos_delta)) * 2.0, -1, 1)
        p_rp = np.clip(np.mean(np.square(np.array([yaw]))) * 1.0, -1, 1)
        r = 0.5 - p_position - p_rp

        self.rew_queue.append([r])
        self.rew_queue.pop(0)
        r_unqueued = self.rew_queue

        # Make neural network observation vector
        obs_list = [*pos_delta, *rotation_rob[1:], *rotation_rob[0:1], *vel_rob, *angular_vel_rob]
        self.obs_queue.append(obs_list)
        self.obs_queue.pop(0)
        obs_raw_unqueued = self.obs_queue

        aux_obs = []
        if self.config["obs_input"] > 0:
            [aux_obs.extend(c) for c in obs_raw_unqueued]
        if self.config["act_input"] > 0:
            [aux_obs.extend(c) for c in act_raw_unqueued]
        if self.config["rew_input"] > 0:
            [aux_obs.extend(c) for c in r_unqueued]

        obs = np.array(aux_obs).astype(np.float32)

        obs_dict = {"euler_rob" : euler_rob, "angular_vel_rob" : angular_vel_rob, "pid_targets" : pid_targets,
                    "velocity_targets" : velocity_targets, "position_rob": position_rob, "pos_delta": pos_delta,
                    "autonomous_control" : autonomous_control}

        return obs, r, False, obs_dict

    def loop_control(self):
        '''
        Target inputs are in radians, throttle is in [0,1]
        Roll, pitch and yaw are in radians. 
        Motor outputs are sent to the motor driver as [0,1]
        '''

        print("Starting the control loop")
        frame_ctr = 0
        slowest_frame = .0001
        obs = np.zeros(self.config["obs_dim"])
        obs_dict = {"euler_rob": [0,0,0], "angular_vel_rob": [0,0,0], "pid_targets": [0,0,0,0],
                    "velocity_targets": [0,0,0,0], "position_rob": [0,0,0], "pos_delta": [0,0,0],
                    "autonomous_control": False}
        while True:
            iteration_starttime = time.time()

            # Calculate stabilization actions
            if obs_dict["autonomous_control"]:
                t_pa_1 = time.time()
                act = self.get_policy_action(obs)
                t_pa_2 = time.time()
                if self.config["time_prints"]: print(f"NN fw pass took: {t_pa_2 - t_pa_1}")
            else:
                act = self.calculate_stabilization_action(obs_dict["euler_rob"], obs_dict["angular_vel_rob"], obs_dict["pid_targets"])

            # Virtual safety net for autonomous control
            if (np.abs(np.array(obs_dict['position_rob'])) > 6).any() and obs_dict["autonomous_control"]:
                print(f"Current position is: {obs_dict['position_rob']} which is outside the safety net, shutting down motors")
                act = 0, 0, 0, 0

            obs, r, done, obs_dict = self.step(act)

            print(f"Pos: {obs_dict['position_rob']}, pos_delta: {obs_dict['pos_delta']}, targets: {obs_dict['targets']}")

            while time.time() - iteration_starttime < self.config["update_period"]: pass
            if time.time() - iteration_starttime > slowest_frame:
                slowest_frame = time.time() - iteration_starttime
            if self.config["time_prints"]: print(f"Slowest frame: {slowest_frame}")

            frame_ctr += 1


    def gather_data(self, n_iterations=20000):

        # Initialize data lists
        data_position = []
        data_vel = []
        data_rotation = []
        data_angular_vel = []
        data_timestamp = []
        data_action = []

        print("Starting the control loop")
        try:
            for i in range(n_iterations):
                iteration_starttime = time.time()

                # Read target control inputs
                throttle, t_yaw, t_roll, t_pitch, autonomous_control = self.JOYStick.get_joystick_input()
                velocity_target = -throttle, -t_roll, -t_pitch, -t_yaw
                pid_target = None

                # Update sensor data
                position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, timestamp = self.AHRS.update()

                # Make neural network observation vector
                obs = np.concatenate((velocity_target, rotation_rob, vel_rob, angular_vel_rob))

                if autonomous_control:
                    m_1, m_2, m_3, m_4 = self.get_policy_action(obs)
                else:
                    m_1, m_2, m_3, m_4 = self.calculate_stabilization_action(euler_rob,
                                                                             [t_roll,t_pitch,t_yaw],
                                                                             throttle)

                data_position.append(position_rob)
                data_vel.append(vel_rob)
                data_rotation.append(rotation_rob)
                data_angular_vel.append(angular_vel_rob)
                data_timestamp.append(timestamp)
                data_action.append([throttle, t_yaw, t_roll, t_pitch])

                # Write control to servos
                self.PWMDriver.write_servos([m_1, m_2, m_3, m_4])

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["update_period"]: pass
        except KeyboardInterrupt:
            print("Interrupted by user")

        # Save data
        prefix = os.path.join("data", time.strftime("%Y_%m_%d"))
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        prefix = prefix + 'quadrotor_'.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))

        data_position = np.array(data_position, dtype=np.float32)
        data_vel = np.array(data_vel, dtype=np.float32)
        data_rotation = np.array(data_rotation, dtype=np.float32)
        data_angular_vel = np.array(data_angular_vel, dtype=np.float32)
        data_timestamp = np.array(data_timestamp, dtype=np.float32)
        data_action = np.array(data_action, dtype=np.float32)

        np.save(prefix + "_position", data_position)
        np.save(prefix + "_vel", data_vel)
        np.save(prefix + "_rotation", data_rotation)
        np.save(prefix + "_angular_vel", data_angular_vel)
        np.save(prefix + "_timestamp", data_timestamp)
        np.save(prefix + "_action", data_action)

        print("Saved data")


if __name__ == "__main__":
    controller = Controller()#
    controller.loop_control()
    
