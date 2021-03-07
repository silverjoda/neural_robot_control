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
        
        self.action_queue = []

        self.turn_transition_thresh = self.config["turn_transition_thresh"]
        self.angle_increment = self.config["angle_increment"]

        self.leg_sensor_gpio_inputs = [11,17,27,10,22,9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)

        self.phases = np.array([-2.0515174865722656, 0.9547860622406006, 1.4069218635559082, -1.3016775846481323, -0.8473407030105591, 2.331017017364502])
        self.x_mult, self.y_offset, self.z_mult, self.z_offset, self.z_lb = [0.06, 0.125, 0.07, -0.12, 0]
        self.dyn_z_array = np.array([self.z_lb] * 6)

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")
        
        # Make IMU
        self.Ahrs = AHRS_RS()
        
        # Load policies
        self.nn_policy_straight = TD3.load("agents/{}".format(self.config["policy_eef"]))

        self.current_nn_policy = self.nn_policy_straight
        self.current_nn_policy_ID = "straight"

        logging.info("Initializing robot hardware")
        self.init_hardware()

        self.observation_timestamp = time.time()
        self.angle = 0
        self.dynamic_time_feature = -1

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

    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            iteration_starttime = time.time()
            # Read joystick
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            #print(turn, vel, button_x)

            # Calculate discrete velocity level
            self.angle_increment = vel * self.config["angle_increment"]
            
            print(turn, vel, button_x)

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                self.hex_write_ctrl([0, -0.5, 0.5] * 6)
                self.Ahrs.reset_yaw()
                self.dynamic_time_feature = -1
                print_sometimes("Idling", 0.1)
                time.sleep(0.1)
            else:
                # Read robot servos and hardware and turn into observation for nn
                clipped_turn = np.clip(-turn, -self.config["turn_clip_value"], self.config["turn_clip_value"])
                target_angles = self.calc_target_angles(clipped_turn)

                # Calculate servo commands from policy action and write to servos
                self.hex_write_ctrl(target_angles)

                self.dynamic_time_feature = np.minimum(self.dynamic_time_feature
                                                       + self.config["dynamic_time_feature_increment"],
                                                       self.config["dynamic_time_feature_ub"])

            while time.time() - iteration_starttime < self.config["update_period"]: pass

    def hex_write_ctrl(self, joint_angles):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

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

    def hex_write_servos_direct(self, act):
        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [act[si-1] for si in scrambled_ids]

        scaled_act = dict(zip(scrambled_ids, scrambled_acts))
        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act)
        return 0

    def calc_target_angles(self, turn):
        contacts = self.read_contacts()

        x_mult_arr = [np.minimum(self.x_mult + turn * self.config["sdev_coeff"], 0.07), np.minimum(self.x_mult - turn * self.config["sdev_coeff"], 0.07)] * 3

        targets = []
        for i in range(6):
            x_cyc = np.sin(self.angle * 2 * np.pi + self.phases[i])
            z_cyc = np.cos(self.angle * 2 * np.pi + self.phases[i])

            target_x = x_cyc * x_mult_arr[i]
            target_y = self.y_offset

            if x_cyc < 0 and z_cyc > 0:
                self.dyn_z_array[i] = self.z_lb

            if contacts[i] < 0:
                self.dyn_z_array[i] = z_cyc
            else:
                self.dyn_z_array[i] = np.maximum(-1, self.dyn_z_array[i] - self.config["z_pressure_coeff"])

            target_z = np.maximum(z_cyc, self.dyn_z_array[i]) * self.z_mult + self.z_offset
            targets.append([target_x, target_y, target_z])

            if not (0.0 > target_z > -0.17):
                print(f"WARNING, TARGET_Z: {target_z}")

        joint_angles = self.my_ikt(targets, self.y_offset)
        return joint_angles

    def hex_get_obs(self, heading_spoof_angle=0):
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

    def read_contacts(self):
        return [GPIO.input(ipt) * 2 - 1 for ipt in self.leg_sensor_gpio_inputs]

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

    def my_ikt(self, target_positions, y_offset, rotation_overlay=None):
        rotation_angles = np.array([np.pi/4,np.pi/4,0,0,-np.pi/4,-np.pi/4])
        if rotation_overlay is not None:
            rotation_angles += rotation_overlay
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], y_offset)
            joint_angles.extend(self.single_leg_ikt(tp_rotated))
        return joint_angles

    def rotate_eef_pos(self, eef_xyz, angle, y_offset):
        return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

    def single_leg_ikt(self, eef_xyz):
        x,y,z = eef_xyz

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        psi = np.arctan(x/y)
        Cx = C * np.sin(psi)
        Cy = C * np.cos(psi)
        R = np.sqrt((x-Cx)**2 + (y-Cy)**2 + (z)**2)
        alpha = np.arcsin(-z/R)

        a = np.arccos((F**2 + R**2 - T**2) / (2 * F * R))
        b = np.arccos((F ** 2 + T ** 2 - R ** 2) / (2 * F * T))

        if np.isnan(a) or np.isnan(b):
            print(a,b)

        assert 0 < a < np.pi or np.isnan(a)
        assert 0 < b < np.pi or np.isnan(b)

        th1 = alpha - q1 - a
        th2 = np.pi - q2 - b

        assert th2 + q2 > 0 or np.isnan(th2)

        return -psi, th1, th2

    def single_leg_dkt(self, angles):
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

    def test_AHRS_RS(self):
        while True:
            roll, pitch, yaw_corrected, quat_yaw_corrected, xd, self.timestamp = self.Ahrs.update()
            print("Roll: {}, Pitch: {}, Yaw: {}, Quat: {}".format(roll, pitch, yaw_corrected, quat_yaw_corrected))
            time.sleep(0.3)

    def test_imu_and_js(self):
        while True:
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            roll, pitch, yaw, quat, timestamp = controller.Ahrs.update(turn)
            print("Turn angle: {}, roll: {}, pitch: {}, yaw: {}, quat: {}".format(turn, roll, pitch, yaw, quat))
            time.sleep(0.3)


if __name__ == "__main__":
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    controller = HexapodController(config)
    #controller.test_AHRS_RS()
    controller.start_ctrl_loop()

