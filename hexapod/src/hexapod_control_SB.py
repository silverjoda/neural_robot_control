import time
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import os
import logging
import smbus
import math
import sys
logging.basicConfig(level=logging.INFO)
import pypot.dynamixel
import pygame
from gyro import IMU
import pyrealsense2 as rs
import math as m
import yaml
import threading
import quaternion
# Torques are positive upwards and when leg is being pushed backward
from stable_baselines import A2C
    

class JoyController():
    def __init__(self):
        logging.info("Initializing joystick controller")
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        turn, vel = [self.joystick.get_axis(3), self.joystick.get_axis(1)]
        button_x = self.joystick.get_button(1)
        pygame.event.clear()

        turn = -turn / 2 # [-0.5, 0.5]
        vel = np.maximum(vel * -1, 0)  # [0, 1]  
        print(f"Turn: {turn}, Vel: {vel}")

        # button_x only when upon press
        if self.button_x_state == 0 and button_x == 1:
            self.button_x_state = 1
            button_x = 1
        elif self.button_x_state == 1 and button_x == 0:
            self.button_x_state = 0
            button_x = 0
        elif self.button_x_state == 1 and button_x == 1:
            self.button_x_state = 1
            button_x = 0
        else:
            self.button_x_state = 0
            button_x = 0

        return turn, vel, button_x


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
    GYRO_SCALER = (2 * np.pi / 360.) * (2000. / (2 ** 15))  # +- 2000 dps across a signed 16 bit value
    ACC_SENSITIVITY = 16384.0

    def __init__(self, config):
        print("Initializing the MPU6050. ")

        self.config = config

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
        self.quat = [0, 0, 0, 1]

        self.gyro_integration_coeff = 0.95
        self.acc_integration_coeff = 1.0 - self.gyro_integration_coeff
        self.gyro_z_deadzone = 0.03

        self.timestamp = time.time()

        self.lock = threading.Lock()

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

    def _update(self):
        while True:
            with self.lock:
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

                self.roll = self.gyro_integration_coeff * \
                            (self.roll + self.gyro_x * dt) + \
                            self.acc_integration_coeff * acc_y_dir
                self.pitch = self.gyro_integration_coeff * \
                             (self.pitch + self.gyro_y * dt) - \
                             self.acc_integration_coeff * acc_x_dir

                if abs(self.gyro_z) > self.gyro_z_deadzone:
                    self.yaw = self.yaw + self.gyro_z * dt

                self.timestamp = t

            time.sleep(self.config["imu_period"])

    def update(self, heading_spoof_angle=0):
        with self.lock:
            yaw_corrected = self.yaw + heading_spoof_angle
            # Correct heading by spoof angle
            self.quat = self.e2q(self.roll, self.pitch, yaw_corrected)
            return self.roll, self.pitch, yaw_corrected, self.quat, 0, self.timestamp

    def reset_yaw(self):
        self.yaw = 0

    def e2q(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return (qx, qy, qz, qw)


class AHRS_RS:
    def __init__(self):
        print("Initializing the rs_t265. ")

        self.rs_to_world_mat = np.array([[0, 1, 0],
                                         [1, 0, 0],
                                         [0, 0, -1]])

        self.pitch_corr_mat = np.array([[0 ,0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]])
        self.pitch_corr_quat = quaternion.from_rotation_matrix(self.pitch_corr_mat)
        
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        self.pipe.start(self.cfg)

        self.yaw_offset = 0
        self.current_heading = 0
        self.timestamp = time.time()

        print("Finished initializing the rs_t265. ")

    def update(self, heading_spoof_angle=0):
        heading_spoof_angle -= self.yaw_offset
        self.timestamp = time.time()

        frames = self.pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()

            # Position
            position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
            vel_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            position_rob = np.matmul(self.rs_to_world_mat, position_rs)
            vel_rob = np.matmul(self.rs_to_world_mat, vel_rs)

            # Rotation: axes are adjusted according how the RS axes are oriented wrt world axes
            rotation_rs_quat = np.quaternion(data.rotation.w, data.rotation.y, data.rotation.x, -data.rotation.z)
            rotation_rob_quat = self.pitch_corr_quat * rotation_rs_quat 
            angular_vel_rob = (data.angular_velocity.y, data.angular_velocity.x, -data.angular_velocity.z)
            roll, pitch, yaw = self.q2e(rotation_rob_quat.x, rotation_rob_quat.y, rotation_rob_quat.z, rotation_rob_quat.w)

            yaw_corrected = yaw + heading_spoof_angle
            quat_yaw_corrected = self.e2q(roll, pitch, yaw_corrected)

            #print("Frame #{}".format(pose.frame_number))
            #print(
            #    "RPY [deg]: Roll: {0:.7f}, Pitch: {1:.7f}, Yaw: {2:.7f}".format(
            #        roll, pitch, yaw
            #    )
            #)
        else:
            position_rob = [0, 0, 0]
            vel_rob = [0, 0, 0]
            rotation_rob_quat = [0, 0, 0, 1]
            angular_vel_rob = [0, 0, 0]
            euler_rob = [0, 0, 0]
            roll, pitch, yaw, yaw_corrected = 0, 0, 0, 0
            quat_yaw_corrected = [0, 0, 0, 1]

        self.current_heading = yaw

        return roll, pitch, yaw_corrected, quat_yaw_corrected, vel_rob[0], self.timestamp

    def q2e(self, x, y, z, w):
        pitch = -m.asin(2.0 * (x * z - w * y))
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        return (roll, pitch, yaw)

    def e2q(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return (qx, qy, qz, qw)

    def reset_yaw(self):
        self.yaw_offset = self.current_heading


class HexapodController:
    def __init__(self, config):
        self.config = config
        self.max_servo_speed = self.config["max_servo_speed"] # [0:1024]
        self.max_servo_torque = self.config["max_servo_torque"]  # [0:1024]
        
        self.action_queue = []
 
        self.joints_rads_low = np.array([-0.5, -1.0, 0.4] * 6)
        self.joints_rads_high = np.array([0.5, 0.4, 1.0] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.joints_10bit_low = ((self.joints_rads_low) / (5.23599) + 0.5) * 1024
        self.joints_10bit_high = ((self.joints_rads_high) / (5.23599) + 0.5) * 1024
        self.joints_10bit_diff = self.joints_10bit_high - self.joints_10bit_low
        
        self.joints_avg = np.zeros((18))
        self.contacts = np.zeros(6) 
        self.contacts_avg = np.zeros(6)
        self.turn_transition_thresh = 0.4
        self.current_discrete_velocity_level = 1

        obs_dim = 24 + 6 * self.config["use_contacts"]

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")
        
        # Make IMU
        self.Ahrs = AHRS_RS()
        
        # Load policies
        self.nn_policy_straight = A2C.load("agents/{}".format(self.config["policy_straight"]))
        self.nn_policy_straight_rough = A2C.load("agents/{}".format(self.config["policy_straight"]))
        self.nn_policy_turn_left = A2C.load("agents/{}".format(self.config["policy_straight"]))
        self.nn_policy_turn_right = A2C.load("agents/{}".format(self.config["policy_straight"]))

        self.current_nn_policy = self.nn_policy_straight
        self.current_nn_policy_ID = "straight"
        self.idling = False

        logging.info("Initializing robot hardware")
        self.init_hardware()


    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            iteration_starttime = time.time()
            # Read joystick
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            #print(turn, vel, button_x)

            # Calculate discrete velocity level
            new_discrete_velocity_level = np.ceil((vel + 0.0001) / 0.34).astype(np.int16)
            if new_discrete_velocity_level != self.current_discrete_velocity_level:
                self.current_discrete_velocity_level = new_discrete_velocity_level
                self.max_servo_speed = self.current_discrete_velocity_level * 100
                speed = dict(zip(self.ids, itertools.repeat(self.max_servo_speed)))
                self.dxl_io.set_moving_speed(speed)
                print("Setting servo speed: {}".format(self.max_servo_speed))
            
            if button_x == 1:
                if self.current_nn_policy_ID == "straight":
                    self.current_nn_policy = self.nn_policy_straight_rough
                    self.current_nn_policy_ID = "straight_rough"
                elif self.current_nn_policy_ID == "straight_rough":
                    self.current_nn_policy = self.nn_policy_straight
                    self.current_nn_policy_ID = "straight"
                print("Switched to {} policy".format(self.current_nn_policy_ID))

            if turn >= self.turn_transition_thresh:
                self.current_nn_policy = self.nn_policy_turn_left
                self.max_servo_speed = 150
                speed = dict(zip(self.ids, itertools.repeat(self.max_servo_speed)))
                self.dxl_io.set_moving_speed(speed)
            if turn <= -self.turn_transition_thresh:
                self.current_nn_policy = self.nn_policy_turn_right
                self.max_servo_speed = 150
                speed = dict(zip(self.ids, itertools.repeat(self.max_servo_speed)))
                self.dxl_io.set_moving_speed(speed)
            if abs(turn) < 0.4:
                if self.current_nn_policy_ID == "straight":
                    self.current_nn_policy = self.nn_policy_straight
                else:
                    self.current_nn_policy = self.nn_policy_straight_rough

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                if not self.idling:
                    self.hex_write_ctrl([0, -1, 0.5] * 6)
                    self.idling = True
                    print("Idling...")
                    time.sleep(0.2)
                    self.Ahrs.reset_yaw()
            elif self.idling:
                self.hex_write_ctrl([0, 0, 0] * 6)
                self.idling = False
                print("Awakening...")
                time.sleep(0.5)

            if not self.idling:
                # Read robot servos and hardware and turn into observation for nn
                policy_obs = self.hex_get_obs()

                # Perform forward pass on nn policy
                policy_act, _ = self.current_nn_policy.predict(policy_obs, deterministic=True)

                # Calculate servo commands from policy action and write to servos
                self.hex_write_ctrl(policy_act)

            if self.config["use_contacts"]:
                self._update_legtip_contact()
                #print(self.contacts)

            while time.time() - iteration_starttime < self.config["update_period"]: pass


    def init_hardware(self):
        '''
        Initialize robot hardware and variables
        :return: Boolean
        '''

        ports = pypot.dynamixel.get_available_ports()
        print('available ports:', ports)

        if not ports:
            raise IOError('No port available.')

        port = ports[0]
        print('Using the first on the list', port)

        self.dxl_io = pypot.dynamixel.DxlIO(port, use_sync_read=False, timeout=0.05, convert=False)
        print('Connected!')

        self.ids = range(1,19)

        scanned_ids = self.dxl_io.scan(self.ids)
        print("Scanned ids: {}".format(scanned_ids))
        assert len(scanned_ids) == len(self.ids)

        self.dxl_io.enable_torque(self.ids)

        speed = dict(zip(self.ids, itertools.repeat(self.max_servo_speed)))
        torque = dict(zip(self.ids, itertools.repeat(self.max_servo_torque)))
        self.dxl_io.set_moving_speed(speed)
        self.dxl_io.set_max_torque(torque)
        self.dxl_io.set_torque_limit(torque)

        # Imu integrated yaw value
        self.yaw = 0

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(dict(zip(self.ids, itertools.repeat(512))))
            time.sleep(2)

    def test_imu_and_js(self):
        while True:
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            roll, pitch, yaw, quat, timestamp = controller.Ahrs.update(turn)
            print("Turn angle: {}, roll: {}, pitch: {}, yaw: {}, quat: {}".format(turn, roll, pitch, yaw, quat))
            time.sleep(0.3)

    def hex_get_obs(self, heading_spoof_angle=0):
        '''
        Read robot hardware and return observation tensor for pytorch
        :return:
        '''
     
        servo_positions = self.dxl_io.get_present_position(self.ids)
                
        # Reverse servo observations
        servo_positions = np.array(servo_positions).astype(np.float32)
        servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1] = 1024 - servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1]  

        # Read IMU (for now spoof perfect orientation)
        roll, pitch, yaw, quat, xd, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)
        print(xd)
        #quat_only_yaw = self.Ahrs.e2q(0, 0, yaw)

        # Turn servo positions into [-1,1] for nn
        obs = ((servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1
        
        # Clip to [-1, 1]
        obs = np.clip(obs, -1, 1)

        # Make nn observation
        obs = np.concatenate((obs, quat, [xd]))
        
        if self.config["use_contacts"]:
            obs = np.concatenate((obs, self.contacts))   
        
        # Add the counter variable at the end
        obs = np.concatenate((obs, [0.0]))

        return obs

    def hex_write_ctrl(self, normalized_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''
    
        nn_act_clipped = np.tanh(normalized_act)

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = np.array([(np.asscalar(nn_act_clipped[i]) * 0.5 + 0.5) * self.joints_10bit_diff[i] + self.joints_10bit_low[i] for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (Thsi part is retarded and will need to be fixed)
        scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1] = 1024 - scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1] 

        scaled_act_dict = dict(zip(self.ids, scaled_act))

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act_dict)

    def hex_write_servos_direct(self, act):
        scaled_act = dict(zip(self.ids, act))
        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act)
        return 0

    def get_normalized_torques(self):
        servo_torques = np.array(self.dxl_io.get_present_load(self.ids))
        torque_dirs = (servo_torques >> 10).astype(np.float32) * 2 - 1
        torque_dirs_corrected = torque_dirs * np.tile([-1, 1, 1, 1, -1, -1], 3)
        raw_torques = (servo_torques % 1024).astype(np.float32)
        normalized_torques = (raw_torques / 1024.) 
        normalized_torques_corrected = normalized_torques * torque_dirs_corrected
        return normalized_torques_corrected

    def _update_legtip_contact(self):
        def _leg_is_in_contact(servo_vec):
            return servo_vec[1] > 0.04 or servo_vec[2] > 0.03

        servo_torques = self.get_normalized_torques()
        for i in range(6):
            if _leg_is_in_contact(servo_torques[i * 3: (i + 1) * 3]):
                self.contacts_avg[i] = np.minimum(self.contacts_avg[i] + 0.55, 1)
                #self.contacts[i] = 1
            else:
                self.contacts_avg[i] = np.maximum(self.contacts_avg[i] - 0.55, -1)
                #self.contacts[i] = -1
        #self.contacs = np.sign(self.contacts_avg)
        self.contacts = np.ones(6)
                
    def test_leg_coordination(self):
        '''
        Perform leg test to determine correct mapping and range
        :return:
        '''

        if not self.config["motors_on"]:
            print("Motors are off, not performing test")

        logging.info("Starting leg coordination test")
        VERBOSE = True
        sc = 1.0
        test_acts = [[0, 0, 0], [0, sc, sc], [0, -sc, -sc], [0, sc, -sc], [0, -sc, sc], [sc, 0, 0], [-sc, 0, 0]]
        for i, a in enumerate(test_acts):
            self.hex_write_ctrl(np.array(a * 6))
            time.sleep(3)
            joints_normed = self.hex_get_obs()[0].detach().numpy()[:18] # rads
            joints_rads = (((joints_normed + 1) / 2) * self.joints_rads_diff) + self.joints_rads_low 
            act_rads = (np.array(a * 6) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low
            
            if VERBOSE:
                print("Obs rads: ", joints_rads) 
                print("Obs norm: ", joints_normed) 
                print("For action rads: ", act_rads) 
                print("action norm: ", a) 
            
        logging.info("Finished leg coordination test, exiting")
        quit()

    def test_AHRS_RS(self):
        while True:
            roll, pitch, yaw_corrected, quat_yaw_corrected, xd, self.timestamp = self.Ahrs.update()
            print("Roll: {}, Pitch: {}, Yaw: {}, Quat: {}".format(roll, pitch, yaw_corrected, quat_yaw_corrected))
            time.sleep(0.3)



if __name__ == "__main__":
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    controller = HexapodController(config)
    #controller.test_AHRS_RS()
    controller.start_ctrl_loop()

