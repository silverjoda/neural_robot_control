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
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import pypot.dynamixel
import pygame
from gyro import IMU
import pyrealsense2 as rs
import math as m
import yaml
import threading
import quaternion
# Torques are positive upwards and when leg is being pushed backward
from stable_baselines3 import A2C, TD3
import RPi.GPIO as GPIO
    
from peripherals import *

class HexapodController:
    def __init__(self, config):
        self.config = config
        self.max_servo_speed = self.config["max_servo_speed"] # [0:1023]
        self.max_servo_torque = self.config["max_servo_torque"]  # [0:1023]
        
        self.action_queue = []

        self.turn_transition_thresh = self.config["turn_transition_thresh"]
        self.angle_increment = self.config["angle_increment"]

        self.leg_sensor_gpio_inputs = [11, 17, 27, 10, 22, 9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)
 
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
        self.nn_policy_straight = TD3.load("agents/{}".format(self.config["policy_straight"]))

        self.current_nn_policy = self.nn_policy_straight
        self.current_nn_policy_ID = "straight"

        logging.info("Initializing robot hardware")
        self.init_hardware()

        self.observation_timestamp = time.time()
        self.previous_joint_angles = [0] * 18
        self.angle = 0
        self.dynamic_time_feature = -1

    def read_contacts(self):
        return [GPIO.input(ipt) for ipt in self.leg_sensor_gpio_inputs]

    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            iteration_starttime = time.time()
            # Read joystick
            turn, vel, button_x = self.joystick_controller.get_joystick_input()
            # print(turn, vel, button_x)

            # Calculate discrete velocity level
            self.angle_increment = vel * self.angle_increment

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                self.hex_write_ctrl([0, -0.5, 0.5] * 6)
                self.Ahrs.reset_yaw()
                self.dynamic_time_feature = -1.
                print_sometimes("Idling",0.1)
                time.sleep(0.1)
            else:
                # Read robot servos and hardware and turn into observation for nn
                policy_obs = self.hex_get_obs(-turn * 3)

                # Perform forward pass on nn policy
                policy_act, _ = self.current_nn_policy.predict(policy_obs, deterministic=True)

                # Calculate servo commands from policy action and write to servos
                self.hex_write_ctrl(policy_act)

                self.dynamic_time_feature = np.minimum(self.dynamic_time_feature + 0.02, 0)

            # while time.time() - iteration_starttime < self.config["update_period"]: pass

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

        # Turn servo positions into [-1,1] for nn
        joints_normed = ((servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1
        joint_angles_rads = (joints_normed * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

        # Torques
        if self.config["read_true_torques"]:
            torques_normed = self.get_normalized_torques()
            torques_nm = torques_normed * 1.5
        else:
            torques_nm = [0] * 18

        # Velocities
        dt = time.time() - self.observation_timestamp
        velocities = (joint_angles_rads - self.previous_joint_angles) / (dt + 1e-5)
        self.previous_joint_angles = joint_angles_rads

        # Contacts
        contacts = self.read_contacts()

        # Make nn observationFalse
        # compiled_obs = torso_quat, torso_vel, [signed_deviation], joint_angles, contacts, [(float(self.step_ctr) / self.config["max_steps"]) * 2 - 1] <- eef
        # compiled_obs = torso_quat, torso_vel, [signed_deviation], time_feature, scaled_joint_angles, contacts, joint_torques, joint_velocities <- This one for wp_* and hexapod configs
        obs = np.concatenate((quat, vel_rob, [yaw], [self.dynamic_time_feature], joint_angles_rads, contacts))
        #obs = np.concatenate((quat, vel_rob, [yaw], [self.dynamic_time_feature], joint_angles_rads, contacts, joint_torques, joint_velocities))

        return obs

    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''
    
        nn_act_clipped = np.tanh(nn_act)

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = np.array([(np.asscalar(nn_act_clipped[i]) * 0.5 + 0.5) * self.joints_10bit_diff[i] + self.joints_10bit_low[i] for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (Thsi part is retarded and will need to be fixed)
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

