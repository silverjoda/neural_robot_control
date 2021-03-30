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
from utils import *

class HexapodController:
    def __init__(self, config):
        self.config = config

        self.phases = np.array(self.config["leg_phases"])
        self.z_mult = self.config["z_mult"]
        self.dyn_z_lb_array = np.array([float(self.config["z_lb"])] * 6)
        self.poc_array = np.array([float(self.config["z_lb"])] * 6)

        self.turn_joints_rads_low = np.array(self.config["turn_joints_rads_low"] * 6)
        self.turn_joints_rads_high = np.array(self.config["turn_joints_rads_high"] * 6)
        self.turn_joints_rads_diff = self.turn_joints_rads_high - self.turn_joints_rads_low

        self.direct_joints_rads_low = np.array(self.config["direct_joints_rads_low"] * 6)
        self.direct_joints_rads_high = np.array(self.config["direct_joints_rads_high"] * 6)
        self.direct_joints_rads_diff = self.direct_joints_rads_high - self.direct_joints_rads_low

        # Load policies
        self.nn_policy_cw = TD3.load("agents/{}".format(self.config["policy_cw"]))
        self.nn_policy_ccw = TD3.load("agents/{}".format(self.config["policy_ccw"]))
        self.nn_policy_direct = TD3.load("agents/{}".format(self.config["policy_direct"]))

        # Make joystick controller
        self.joystick_controller = JoyController()
        logging.info("Loading policies: ")
        
        # Make IMU
        self.Ahrs = AHRS_RS()

        logging.info("Initializing robot hardware")
        self.init_hardware()

        self.observation_timestamp = time.time()
        self.angle = 0
        self.dynamic_step_ctr = 0
        self.current_servo_speed = 0
        self.idling = True
        self.xd_queue = []
        self.prev_act = np.zeros(18)

    def init_hardware(self):
        # Set GPIO for sensor inputs
        GPIO.setmode(GPIO.BCM)
        for ipt in self.config["leg_sensor_gpio_inputs"]:
            GPIO.setup(ipt, GPIO.IN)

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

        speed = dict(zip(self.ids, itertools.repeat(200)))
        torque = dict(zip(self.ids, itertools.repeat(900)))
        self.dxl_io.set_moving_speed(speed)
        self.dxl_io.set_max_torque(torque)
        self.dxl_io.set_torque_limit(torque)

        # Imu integrated yaw value
        self.yaw = 0

        if self.config["motors_on"]:
            tar_pos = [512, 100, 1023 - 200, 512, 1023 - 100, 200] * 3
            self.dxl_io.set_goal_position(dict(zip(self.ids, tar_pos)))
            time.sleep(1)

            print("MOTORS ON")

    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            iteration_starttime = time.time()
            # Read joystick
            turn, vel, height, button_x, button_x_event = self.joystick_controller.get_joystick_input()

            self.z_mult = self.z_mult_static # + height * 0.03

            # Calculate discrete velocity level
            self.angle_increment = vel * self.config["angle_increment"]

            # TMP DEBUG
            # if button_x_event:
            #     self.Ahrs.reset_yaw()
            #     self.Ahrs.reset_relative_position()
            # if button_x:
            #     self.Ahrs.update()
            #     pos_rob_relative, vel_rob_relative = self.Ahrs.get_relative_position_and_velocity()
            #     print(self.Ahrs.position_rob, pos_rob_relative, vel_rob_relative)

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                self.hex_write_max_torques(0)
                self.Ahrs.reset_yaw()
                self.Ahrs.reset_relative_position()
                self.dynamic_step_ctr = 0
                self.xd_queue = []
                self.idling = True
                print_sometimes("Idling", 0.1)
                time.sleep(0.1)
            else:
                if self.idling:
                    self.hex_write_max_torques(self.max_servo_torque)
                    self.idling = False
                    print("Active")

                # Read robot servos and hardware and turn into observation for nn
                clipped_turn = -turn

                if abs(clipped_turn) > 0.47:
                    self.hex_write_servo_speed(self.max_servo_speed * np.maximum(vel, 0.3))
                    self.current_servo_speed = vel

                    policy_obs = self.hex_get_obs_turn(clipped_turn)
                    if clipped_turn > 0:
                        policy_act, _ = self.nn_policy_cw.predict(policy_obs, deterministic=True)
                    else:
                        policy_act, _ = self.nn_policy_ccw.predict(policy_obs, deterministic=True)
                    self.hex_write_ctrl_nn(policy_act, mode="turn")
                else:
                    if self.current_servo_speed < 0.95:
                        self.current_servo_speed = 1.0
                        self.hex_write_servo_speed(self.max_servo_speed)

                    if button_x_event:
                        self.Ahrs.reset_yaw()
                        self.Ahrs.reset_relative_position()
                        self.dynamic_step_ctr = 0
                        self.xd_queue = []

                    if button_x:
                        policy_obs = self.hex_get_obs_direct(clipped_turn)
                        policy_act, _ = self.nn_policy_direct.predict(policy_obs, deterministic=True)
                        self.hex_write_ctrl_nn(policy_act, mode="direct")
                        self.dynamic_step_ctr = np.minimum(self.dynamic_step_ctr + 1, self.config["dynamic_max_steps"])
                    else:
                        target_angles = self.calc_target_joint_angles_cyc(clipped_turn)
                        self.hex_write_ctrl(target_angles)

            while time.time() - iteration_starttime < self.config["update_period"]: pass

    def hex_write_ctrl(self, joint_angles):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        joint_angles_servo = rads_to_servo(joint_angles)
        joint_angles_servo = np.array([np.asscalar(joint_angles_servo[i]) for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        joint_angles_servo[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - joint_angles_servo[np.array([4,5,6,10,11,12,16,17,18])-1]

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [joint_angles_servo[si - 1] for si in scrambled_ids]
        scaled_act_dict = dict(zip(scrambled_ids, scrambled_acts))

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act_dict)

    def hex_write_ctrl_nn(self, nn_act, mode="direct"):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        target_joints_norm = np.tanh(nn_act)
        self.prev_act = target_joints_norm

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = None
        if mode == "direct":
            target_joints_rads = norm_to_rads(target_joints_norm, self.direct_joints_rads_low, self.direct_joints_rads_diff)
        if mode == "turn":
            target_joints_rads = norm_to_rads(target_joints_norm, self.turn_joints_rads_low, self.turn_joints_rads_diff)
        assert target_joints_rads is not None

        target_joints_servo = rads_to_servo(target_joints_rads)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        target_joints_servo[np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1] = 1023 - target_joints_servo[
            np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1]

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [target_joints_servo[si - 1] for si in scrambled_ids]
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

    def hex_write_max_torques(self, max_torque):
        torque_dict = dict(zip(self.ids, itertools.repeat(max_torque)))
        self.dxl_io.set_max_torque(torque_dict)
        self.dxl_io.set_torque_limit(torque_dict)

    def hex_write_servo_speed(self, servo_speed):
        speed_dict = dict(zip(self.ids, itertools.repeat(int(servo_speed))))
        self.dxl_io.set_moving_speed(speed_dict)

    def calc_target_joint_angles_cyc(self, turn):
        contacts = read_contacts()

        x_mult_arr = [np.minimum(self.config["x_mult"] + turn * self.config["turn_coeff"], 0.08), np.minimum(self.config["x_mult"] - turn * self.config["turn_coeff"], 0.08)] * 3

        targets = []
        for i in range(6):
            x_cyc = np.sin(self.angle * 2 * np.pi + self.phases[i])
            z_cyc = np.cos(self.angle * 2 * np.pi + self.phases[i])

            target_x = x_cyc * x_mult_arr[i]
            target_y = self.config["y_offset"]

            if x_cyc < 0 and z_cyc > 0.0:
                self.dyn_z_lb_array[i] = self.config["z_lb"]

            if contacts[i] < 0:
                self.dyn_z_lb_array[i] = z_cyc
                self.poc_array[i] = 1
            else:
                if self.poc_array[i] == 1:
                    self.poc_array[i] = z_cyc
                self.dyn_z_lb_array[i] = self.poc_array[i] - self.config["z_pressure_coeff"]

            target_z = np.maximum(z_cyc, self.dyn_z_lb_array[i]) * self.z_mult + self.config["z_offset"]
            targets.append([target_x, target_y, target_z])

        joint_angles = my_ikt(targets, self.config["y_offset"])
        self.angle += self.angle_increment

        return joint_angles

    def hex_get_obs_turn(self, heading_spoof_angle=0):
        servo_positions, joints_rads, joints_norm = self.read_joint_angles(self.turn_joints_rads_low, self.turn_joints_rads_diff)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)
        obs = np.concatenate((quat, vel_rob, [yaw], [0], joints_norm))

        return obs

    def hex_get_obs_direct(self, heading_spoof_angle=0):
        servo_positions, joints_rads, joints_norm = self.read_joint_angles(self.direct_joints_rads_low, self.direct_joints_rads_diff)
        roll, pitch, yaw, quat, vel_rob, timestamp = self.Ahrs.update(heading_spoof_angle=heading_spoof_angle)
        pos_rob_relative, vel_rob_relative = self.Ahrs.get_relative_position_and_velocity()
        xd, yd, zd = vel_rob_relative

        # Avg vel
        self.xd_queue.append(xd)
        if len(self.xd_queue) > 15:
            del self.xd_queue[0]
        avg_vel = sum(self.xd_queue) / len(self.xd_queue)
        avg_vel = avg_vel / 0.15 - 1

        # Torques
        if self.config["velocities_and_torques"]:
            torques_norm = self.get_normalized_torques()
            joint_torques = torques_norm * 1.5
        else:
            joint_torques = [0] * 18

        joint_velocities = [0] * 18

        self.dynamic_time_feature = (float(self.dynamic_step_ctr) / self.config["dynamic_max_steps"]) * 2 - 1

        if self.config["velocities_and_torques"]:
            obs = np.concatenate((quat, vel_rob_relative, pos_rob_relative, [yaw], [self.dynamic_time_feature], [avg_vel], joints_norm, joint_torques, joint_velocities))
        else:
            # torso_quat, torso_vel, torso_pos, [signed_deviation], time_feature, [avg_vel], scaled_joint_angles, self.prev_act
            obs = np.concatenate((quat, vel_rob_relative, pos_rob_relative, [yaw], [self.dynamic_time_feature], [avg_vel], joints_norm, self.prev_act))

        # TODO: Print entire obs with labels for debugging purposes
        # TODO: Make action smoothing
        #obs_dict = {"Quat" : quat, "vel_rob_relative" : vel_rob_relative, "pos_rob_relative" : pos_rob_relative, "yaw" : yaw, "dynamic_time_feature" : self.dynamic_time_feature,
        #            "avg_vel" : avg_vel, "joints_normed" : joints_normed, "prev_act" : self.prev_act}

        return obs

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

    def read_joint_angles(self, low, diff):
        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_servo_positions = self.dxl_io.get_present_position(scrambled_ids)
        servo_positions = [scrambled_servo_positions[scrambled_ids.index(i + 1)] for i in range(18)]

        # Reverse servo observations
        servo_positions = np.array(servo_positions).astype(np.float32)
        servo_positions[np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1] = 1023 - servo_positions[
            np.array([4, 5, 6, 10, 11, 12, 16, 17, 18]) - 1]

        # Turn servo positions into [-1,1] for nn
        joints_rads = servo_to_rads(servo_positions)
        joints_normed = rads_to_norm(joints_rads, low, diff)

        return servo_positions, joints_rads, joints_normed

    def test_joint_angles(self):
        test_angles_1 = [-1] * 18
        test_angles_2 = [0] * 18
        test_angles_3 = [1] * 18

        # TODO: Continue here.. Make proper norm->rads->servo and the other way around and test thoroughly to see if reads match writes, etc
        # TODO: Check if for give observation, neural network in simulation gives same result as in robot
        # TODO: Make action smoothing .


if __name__ == "__main__":
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    controller = HexapodController(config)
    #controller.test_AHRS_RS()
    controller.start_ctrl_loop()

