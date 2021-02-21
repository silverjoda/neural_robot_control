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
from stable_baselines3 import A2C
import RPi.GPIO as GPIO
import pybullet as p
    
from peripherals import *

class HexapodController:
    def __init__(self, config):
        self.config = config
        self.max_servo_speed = self.config["max_servo_speed"] # [0:1024]
        self.max_servo_torque = self.config["max_servo_torque"]  # [0:1024]
        
        self.action_queue = []

        # TODO: remove this
        self.joints_rads_low = np.array([-0.5, -1.0, 0.4] * 6)
        self.joints_rads_high = np.array([0.5, 0.4, 1.0] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.joints_10bit_low = (self.joints_rads_low / 5.23599 + 0.5) * 1024
        self.joints_10bit_high = ((self.joints_rads_high) / (5.23599) + 0.5) * 1024
        self.joints_10bit_diff = self.joints_10bit_high - self.joints_10bit_low

        self.turn_transition_thresh = 0.4
        self.angle = 0
        self.angle_increment = self.config["angle_increment"]

        self.leg_sensor_gpio_inputs = [11,17,27,10,22,9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)

        self.phases_op = np.array(
            [0.010952126234769821, 2.5668561458587646, -1.7436176538467407, 0.7150714993476868, 2.0461928844451904,
             -0.8317734599113464])
        self.current_phases = self.phases_op
        self.x_mult, self.y_offset, self.z_mult, self.z_offset, self.phase_offset_l, self.phase_offset_r = [
            0.06,
            0.15,
            0.03,
            -0.09,
            0.33068275451660156,
            0.41586756706237793]

        self.urdf_name = self.config["urdf_name"]
        self.client_ID = p.connect(p.DIRECT)
        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name),
                           physicsClientId=self.client_ID)

        self.left_offset, self.right_offset = np.array([self.phase_offset_l, self.phase_offset_r])

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")
        
        # Make IMU
        self.Ahrs = AHRS_RS()
        
        # Load policies
        self.nn_policy_straight = A2C.load("agents/{}".format(self.config["policy_straight"]))

        self.current_nn_policy = self.nn_policy_straight
        self.current_nn_policy_ID = "straight"
        self.idling = False

        logging.info("Initializing robot hardware")
        self.init_hardware()

        self.observation_timestamp = time.time()
        self.previous_joint_angles = [0] * 18

    def read_contacts(self):
        return [GPIO.input(ipt) for ipt in self.leg_sensor_gpio_inputs]

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
                self.angle_increment = vel * 0.035
                #self.dxl_io.set_moving_speed(speed)
                print("Setting servo speed: {}".format(self.max_servo_speed))
            
            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                if not self.idling:
                    self.hex_write_ctrl([0, -0.5, 0.5] * 6)
                    self.idling = True
                    print("Idling...")
                    time.sleep(0.2)
                    self.Ahrs.reset_yaw()
            elif self.idling:
                self.hex_write_ctrl([0, 0, 0] * 6)
                self.idling = False
                print("Awakening...")
                time.sleep(0.2)
                self.Ahrs.reset_yaw()
            
            if not self.idling:
                # Read robot servos and hardware and turn into observation for nn
                policy_obs = self.hex_get_obs(-turn * 3)

                # Perform forward pass on nn policy
                policy_act, _ = self.current_nn_policy.predict(policy_obs, deterministic=True)

                # Calculate servo commands from policy action and write to servos
                self.hex_write_ctrl(policy_act)


            #while time.time() - iteration_starttime < self.config["update_period"]: pass


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
        servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1]

        # Read IMU (for now spoof perfect orientation)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)

        # Turn servo positions into [-1,1] for nn
        joints_normed = (servo_positions / 1023) * 2 - 1
        joint_angles_rads = joints_normed * 2.618

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

        # Make nn observation
        # compiled_obs = torso_quat, torso_vel, [signed_deviation], joint_angles, contacts, [(float(self.step_ctr) / self.config["max_steps"]) * 2 - 1] <- eef
        # compiled_obs = torso_quat, torso_vel, [signed_deviation], time_feature, scaled_joint_angles, contacts, joint_torques, joint_velocities <- This one for wp_* and hexapod configs
        obs = np.concatenate((quat, vel_rob, [yaw], joint_angles_rads, contacts, [0]))

        return obs

    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        ctrl_raw = np.tanh(nn_act)

        self.current_phases = self.phases_op + np.tanh(ctrl_raw[0:6]) * np.pi * self.config["phase_scalar"]
        self.left_offset, self.right_offset = np.array([self.phase_offset_l, self.phase_offset_r]) + np.tanh(
            ctrl_raw[6:8]) * np.pi * self.config["phase_offset_scalar"]

        targets = []
        for i in range(6):
            target_x = np.cos(-self.angle * 2 * np.pi + self.phases_op[i]) * self.x_mult
            target_y = self.y_offset
            target_z = np.clip(np.sin(
                -self.angle * 2 * np.pi + self.phases_op[i] + self.left_offset * bool(i % 2) + self.right_offset * bool(
                    (i + 1) % 2)) * self.z_mult + self.z_offset + np.tanh(ctrl_raw[8 + i]) * self.config[
                                   "z_aux_scalar"], -0.12, -0.03)
            targets.append([target_x, target_y, target_z])

        joint_angles = self.my_ikt(targets, self.y_offset)
        joint_angles_normed = np.clip(np.array(joint_angles) / 2.618, -1, 1) # [-1,1] corresponding to bounds on servos
        joint_angles_servo = (joint_angles_normed * 0.5 + 0.5) * 1023

        self.angle += self.angle_increment

        scaled_act = np.array([np.asscalar(joint_angles_servo[i]) for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1]
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

