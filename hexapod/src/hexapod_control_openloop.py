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

class FF_HEX_EEF(nn.Module):
    def __init__(self):
        super(FF_HEX_EEF, self).__init__()

    def forward(self, _):
        #0.17186029255390167, 0.6840080618858337, 2.2731125354766846, -2.1205337047576904, -0.19777145981788635, 1.6162649393081665, -2.815896987915039, 0.1488673835992813]
        clipped_params = [0.06, # x_mult
                          0.15, # y_offset
                          0.03, # z_mult
                          -0.09, # z_offset
                          0.17186029255390167, # phase_offset_l
                          0.684008061, # phase_offset_r
                          2.273112,
                          -2.1205337047576904,
                          -0.197771,
                          1.616264,
                          -2.81589,
                          0.1488673]

        return clipped_params

    def sample_action(self, _):
        with T.no_grad():
            act = self.forward(None)
        return act

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
        button_x = self.joystick.get_button(0)
        pygame.event.clear()

        turn = -turn / 2 # [-0.5, 0.5]
        vel = np.maximum(vel * -1, 0)  # [0, 1]  
        #print(f"Turn: {turn}, Vel: {vel}, Button: {button_x}")

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


class HexapodController:
    def __init__(self, config):
        self.config = config
        self.max_servo_speed = self.config["max_servo_speed"] # [0:1024]
        self.max_servo_torque = self.config["max_servo_torque"]  # [0:1024]

        self.angle = 0
        self.angle_increment = 0.001

        self.leg_sensor_gpio_inputs = [11,17,27,10,22,9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)

        self.current_discrete_velocity_level = 1

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")

        # Load policies
        self.nn_policy_straight = FF_HEX_EEF()

        self.current_nn_policy = self.nn_policy_straight
        self.current_nn_policy_ID = "straight"
        self.idling = False

        logging.info("Initializing robot hardware")
        self.init_hardware()

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
                    self.hex_write_ctrl([0, -1, 0.5] * 6)
                    self.idling = True
                    print("Idling...")
                    time.sleep(0.2)

            elif self.idling:
                self.hex_write_ctrl([0, 0, 0] * 6)
                self.idling = False
                print("Awakening...")
                time.sleep(0.2)

            
            if not self.idling:
                # Perform forward pass on nn policy
                policy_act = self.current_nn_policy(None)

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
        servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1] = 1024 - servo_positions[np.array([4,5,6,10,11,12,16,17,18])-1]

        # Read IMU (for now spoof perfect orientation)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)
        contacts = self.read_contacts()
        print(yaw)

        # Turn servo positions into [-1,1] for nn
        joints_normed = ((servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1
        joint_angles_rads = (joints_normed * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

        # Make nn observationFalse
        # compiled_obs = torso_quat, torso_vel, [signed_deviation, (self.angle % (np.pi * 2) - np.pi)], joint_angles, current_phases_obs, offset_obs, contacts
        obs = np.concatenate((quat, vel_rob, [yaw, 0], joint_angles_rads, [0] * 6, [0,0], contacts, [0]))

        return obs

    def hex_write_ctrl(self, normalized_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''
    
        x_mult, y_offset, z_mult, z_offset, phase_offset_l, phase_offset_r, *phases = [0.0600, 0.1500, 0.0300, -0.0900, 0.1719, 0.6840, 2.2731, -2.1205, -0.1978, 1.6163, -2.8159, 0.1489]

        targets = []
        for i in range(6):
            target_x = np.cos(-self.angle * 2 * np.pi + phases[i]) * x_mult
            target_y = y_offset
            target_z = np.sin(
                -self.angle * 2 * np.pi + phases[i] + phase_offset_l * bool(i % 2) + phase_offset_r * bool(
                    (i + 1) % 2)) * z_mult + z_offset
            targets.append([target_x, target_y, target_z])

        joint_angles = self.my_ikt(targets, y_offset)

        self.angle += self.angle_increment

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = np.array([(np.asscalar(joint_angles[i]) * 0.5 + 0.5) * self.joints_10bit_diff[i] + self.joints_10bit_low[i] for i in range(18)]).astype(np.uint16)

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

    def my_ikt(self, target_positions, y_offset):
        rotation_angles = [np.pi/4,np.pi/4,0,0,-np.pi/4,-np.pi/4]
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

    def single_leg_dkt(self, angles):
        psi, th1, th2 = angles

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        Ez = -F * np.sin(q1 + th1) - T * np.sin(q1 + th1 + q2 + th2)
        Ey = C + F * np.cos(q1 + th1) + T * np.cos(q1 + th1 + q2 + th2)
        Ex = Ey * np.sin(psi)

        return (Ex, Ey, Ez)

if __name__ == "__main__":
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    controller = HexapodController(config)
    #controller.test_AHRS_RS()
    controller.start_ctrl_loop()

