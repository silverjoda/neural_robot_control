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
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import pypot.dynamixel
import pygame
from gyro import IMU
import pyrealsense2 as rs
import math as m
import yaml
import threading
import quaternion
# Torques are positive upwards and when leg is being pushed backward
from stable_baselines3 import TD3
import RPi.GPIO as GPIO
    
from peripherals import *

class HexapodController:
    def __init__(self, config):
        self.config = config
        self.max_servo_speed = self.config["max_servo_speed"] # [0:1024]
        self.max_servo_torque = self.config["max_servo_torque"]  # [0:1024]

        self.turn_transition_thresh = self.config["turn_transition_thresh"]
        self.angle_increment = self.config["angle_increment"]

        self.leg_sensor_gpio_inputs = [11, 17, 27, 10, 22, 9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)

        self.phases = np.array([-4.280901908874512, 5.452933311462402, -0.7993605136871338, 2.3967010974884033, 2.4376134872436523, -0.6086690425872803])
        self.x_mult, self.y_offset, self.z_mult, self.z_offset = [
            0.06,
            0.12,
            0.03,
            -0.12]

        self.joints_rads_low = np.array(self.config["joints_rads_low"] * 6)
        self.joints_rads_high = np.array(self.config["joints_rads_high"] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.joints_10bit_low = ((self.joints_rads_low) / (5.23599) + 0.5) * 1024
        self.joints_10bit_high = ((self.joints_rads_high) / (5.23599) + 0.5) * 1024
        self.joints_10bit_diff = self.joints_10bit_high - self.joints_10bit_low

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")
        
        # Make IMU
        self.Ahrs = AHRS_RS()
        
        # Load policies
        self.nn_policy_eef = TD3.load("agents/{}".format(self.config["policy_eef"]))
        self.nn_policy_direct = TD3.load("agents/{}".format(self.config["policy_direct"]))

        self.control_modes = ["eef", "direct"]
        self.current_control_mode_idx = 0

        logging.info("Initializing robot hardware")
        self.init_hardware()

        self.observation_timestamp = time.time()
        self.angle = 0
        self.dynamic_time_feature = -1
        self.xd_queue = []

    def read_contacts(self):
        return [GPIO.input(ipt) * 2 - 1 for ipt in self.leg_sensor_gpio_inputs]

    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            iteration_starttime = time.time()
            # Read joystick
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            #print(turn, vel, button_x)

            # Calculate discrete velocity level
            self.angle_increment = vel * self.config["angle_increment"]

            if button_x:
                self.current_control_mode_idx = not self.current_control_mode_idx

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                self.hex_write_ctrl_direct([0, -0.5, 0.5] * 6)
                self.Ahrs.reset_yaw()
                self.dynamic_time_feature = -1
                print_sometimes("Idling", 0.1)
                time.sleep(0.1)
            else:
                # Read robot servos and hardware and turn into observation for nn
                clipped_turn = np.clip(-turn * 2, -self.config["turn_clip_value"], self.config["turn_clip_value"])

                if self.control_modes[self.current_control_mode_idx] == "direct":
                    speed = dict(zip(self.ids, itertools.repeat(int(self.max_servo_speed * vel))))
                    self.dxl_io.set_moving_speed(speed)

                    policy_obs = self.hex_get_obs_direct(clipped_turn)
                    policy_act, _ = self.nn_policy_direct.predict(policy_obs, deterministic=True)
                    self.hex_write_ctrl_direct(policy_act)
                else:
                    policy_obs = self.hex_get_obs_eef(clipped_turn)
                    policy_act, _ = self.nn_policy_eef.predict(policy_obs, deterministic=True)
                    self.hex_write_ctrl_eef(policy_act)

                self.dynamic_time_feature = np.minimum(self.dynamic_time_feature
                                                       + self.config["dynamic_time_feature_increment"],
                                                       self.config["dynamic_time_feature_ub"])

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

    def hex_get_obs_eef(self, heading_spoof_angle=0):
        '''
        Read robot hardware and return observation tensor for pytorch
        :return:
        '''

        scrambled_ids = list(range(1,19))
        np.random.shuffle(scrambled_ids)
        scrambled_servo_positions = self.dxl_io.get_present_position(scrambled_ids)
        servo_positions = [scrambled_servo_positions[scrambled_ids.index(i+1)] for i in range(18)]

        # Reverse servo observations
        servo_positions = np.array(servo_positions).astype(np.float32)
        servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1]

        # Read IMU (for now spoof perfect orientation)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)


        # Turn servo positions into rads
        joints_normed = (servo_positions / 1023) * 2 - 1
        joint_angles_rads = joints_normed * 2.618

        # Contacts
        contacts = self.read_contacts()

        # Make nn observation
        obs = np.concatenate((quat, vel_rob, [yaw], joint_angles_rads, contacts, [self.dynamic_time_feature]))

        return obs

    def hex_get_obs_direct(self, heading_spoof_angle=0):
        '''
        Read robot hardware and return observation tensor for pytorch
        :return:
        '''

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_servo_positions = self.dxl_io.get_present_position(scrambled_ids)
        servo_positions = [scrambled_servo_positions[scrambled_ids.index(i + 1)] for i in range(18)]

        # Reverse servo observations
        servo_positions = np.array(servo_positions).astype(np.float32)
        servo_positions[np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1] = 1023 - servo_positions[
            np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1]

        # Read IMU (for now spoof perfect orientation)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)
        xd, yd, zd = vel_rob

        # Avg vel
        self.xd_queue.append(xd)
        if len(self.xd_queue) > 15:
            del self.xd_queue[0]
        avg_vel = sum(self.xd_queue) / len(self.xd_queue)
        avg_vel = avg_vel / 0.15 - 1

        # Turn servo positions into [-1,1] for nn
        joints_normed = ((servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1
        joint_angles_rads = (joints_normed * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

        # Torques
        if self.config["velocities_and_torques"]:
            torques_normed = self.get_normalized_torques()
            joint_torques = torques_normed * 1.5
        else:
            joint_torques = [0] * 18

        # Velocities
        # dt = time.time() - self.observation_timestamp
        joint_velocities = [0] * 18#(joint_angles_rads - self.previous_joint_angles) / (dt + 1e-5)
        # self.previous_joint_angles = joint_angles_rads

        if not self.config["velocities_and_torques"]:
            obs = np.concatenate((quat, vel_rob, [yaw], [self.dynamic_time_feature], [avg_vel], joints_normed))
        else:
            obs = np.concatenate((quat, vel_rob, [yaw], [self.dynamic_time_feature], [avg_vel], joint_angles_rads, joint_torques, joint_velocities))

        return obs

    def hex_write_ctrl_eef(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        ctrl_raw = np.tanh(nn_act)

        x_mult_arr = [self.x_mult + np.tanh(ctrl_raw[7]) * self.config["x_mult_scalar"],
                      self.x_mult + np.tanh(ctrl_raw[8]) * self.config["x_mult_scalar"]] * 3

        targets = []
        for i in range(6):
            target_x = np.cos(-self.angle * 2 * np.pi + self.phases_op[i]) * x_mult_arr[i]
            target_y = self.y_offset
            target_z = np.clip(
                np.sin(-self.angle * 2 * np.pi + self.phases_op[i]) * self.z_mult + self.z_offset + np.tanh(
                    ctrl_raw[8 + i]) * self.config["z_aux_scalar"], -0.13, -0.04)
            targets.append([target_x, target_y, target_z])

        joint_angles = self.my_ikt(targets, self.y_offset)
        joint_angles_normed = np.clip(np.array(joint_angles) / 2.618, -1, 1) # [-1,1] corresponding to bounds on servos
        joint_angles_servo = (joint_angles_normed * 0.5 + 0.5) * 1023

        self.angle += self.angle_increment

        scaled_act = np.array([np.asscalar(joint_angles_servo[i]) for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1]

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [scaled_act[si - 1] for si in scrambled_ids]
        scaled_act_dict = dict(zip(scrambled_ids, scrambled_acts))

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act_dict)

    def hex_write_ctrl_direct(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        nn_act_clipped = np.tanh(nn_act)

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = np.array(
            [(np.asscalar(nn_act_clipped[i]) * 0.5 + 0.5) * self.joints_10bit_diff[i] + self.joints_10bit_low[i] for i
             in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        scaled_act[np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1] = 1023 - scaled_act[
            np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1]

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [scaled_act[si - 1] for si in scrambled_ids]
        scaled_act_dict = dict(zip(scrambled_ids, scrambled_acts))

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act_dict)

    def hex_write_servos_direct(self, act):
        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [act[si-1] for si in scrambled_ids]

        scaled_act = dict(zip(scrambled_ids, scrambled_acts))
        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act)
        return 0

    def get_normalized_torques(self):
        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_servo_torques = np.array(self.dxl_io.get_present_load(self.ids))
        servo_torques = np.array([scrambled_servo_torques[scrambled_ids.index(i + 1)] for i in range(18)])

        torque_dirs = (servo_torques >> 10).astype(np.float32) * 2 - 1
        torque_dirs_corrected = torque_dirs * np.tile([-1, 1, 1, 1, -1, -1], 3)
        raw_torques = (servo_torques % 1023).astype(np.float32)
        normalized_torques = (raw_torques / 1023.)
        normalized_torques_corrected = normalized_torques * torque_dirs_corrected
        return normalized_torques_corrected

    def my_ikt(self, target_positions, y_offset):
        rotation_angles = [np.pi / 4, np.pi / 4, 0, 0, -np.pi / 4, -np.pi / 4]
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], y_offset)
            joint_angles.extend(self.single_leg_ikt(tp_rotated))
        return joint_angles

    def rotate_eef_pos(self, eef_xyz, angle, y_offset):
        return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

    def single_leg_ikt(self, eef_xyz):
        x,y,z = eef_xyz

        assert -0.15 < x < 0.15
        assert 0.05 < y < 0.3
        assert -0.2 < z < 0.2

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        psi = (np.arcsin(x/y))
        Cx = C * np.sin(psi)
        Cy = C * np.cos(psi)
        R = np.sqrt((x-Cx)**2 + (y-Cy)**2 + (z)**2)
        alpha = np.arcsin(-z/R)

        a = np.arccos((F**2 + R**2 - T**2) / (2 * F * R))
        b = np.arccos((F ** 2 + T ** 2 - R ** 2) / (2 * F * T))

        th1 = alpha - q1 - a
        th2 = np.pi - q2 - b

        return -psi, th1, th2

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

