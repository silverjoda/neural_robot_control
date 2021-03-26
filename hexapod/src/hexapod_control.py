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

        self.leg_sensor_gpio_inputs = [11,17,27,10,22,9]

        GPIO.setmode(GPIO.BCM)
        for ipt in self.leg_sensor_gpio_inputs:
            GPIO.setup(ipt, GPIO.IN)

        self.phases = np.array([-4.280901908874512, 5.452933311462402, -0.7993605136871338, 2.3967010974884033, 2.4376134872436523, -0.6086690425872803])
        self.x_mult, self.y_offset, self.z_mult_static, self.z_offset, self.z_lb = [self.config["x_mult"], self.config["y_offset"], self.config["z_mult"], self.config["z_offset"], self.config["z_lb"]]
        self.z_mult = self.z_mult_static
        self.dyn_z_lb_array = np.array([float(self.z_lb)] * 6)
        self.poc_array = np.array([float(self.z_lb)] * 6)

        self.turn_joints_rads_low = np.array(self.config["turn_joints_rads_low"] * 6)
        self.turn_joints_rads_high = np.array(self.config["turn_joints_rads_high"] * 6)
        self.turn_joints_rads_diff = self.turn_joints_rads_high - self.turn_joints_rads_low

        self.turn_joints_10bit_low = ((self.turn_joints_rads_low) / (5.23599) + 0.5) * 1024
        self.turn_joints_10bit_high = ((self.turn_joints_rads_high) / (5.23599) + 0.5) * 1024
        self.turn_joints_10bit_diff = self.turn_joints_10bit_high - self.turn_joints_10bit_low

        self.direct_joints_rads_low = np.array(self.config["direct_joints_rads_low"] * 6)
        self.direct_joints_rads_high = np.array(self.config["direct_joints_rads_high"] * 6)
        self.direct_joints_rads_diff = self.direct_joints_rads_high - self.direct_joints_rads_low

        self.direct_joints_10bit_low = ((self.direct_joints_rads_low) / (5.23599) + 0.5) * 1024
        self.direct_joints_10bit_high = ((self.direct_joints_rads_high) / (5.23599) + 0.5) * 1024
        self.direct_joints_10bit_diff = self.direct_joints_10bit_high - self.direct_joints_10bit_low

        # Load policies
        self.nn_policy_cw = TD3.load("agents/{}".format(self.config["policy_cw"]))
        self.nn_policy_ccw = TD3.load("agents/{}".format(self.config["policy_ccw"]))
        self.nn_policy_direct = TD3.load("agents/{}".format(self.config["policy_direct"]))

        self.control_modes = ["cyc", "direct"]
        self.current_control_mode_idx = 0

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
        self.dyn_speed = 0
        self.idling = True
        self.xd_queue = []
        self.prev_act = np.zeros(18)

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
            turn, vel, height, button_x, button_x_event = self.joystick_controller.get_joystick_input()

            self.z_mult = self.z_mult_static + height * 0.03

            # Calculate discrete velocity level
            self.angle_increment = vel * self.config["angle_increment"]

            if button_x:
                self.current_control_mode_idx = not self.current_control_mode_idx
                if self.current_control_mode_idx == 1:
                    self.Ahrs.reset_relative_position()

            # Idle
            if vel < 0.1 and abs(turn) < 0.1:
                torque = dict(zip(self.ids, itertools.repeat(0)))
                self.dxl_io.set_max_torque(torque)
                self.dxl_io.set_torque_limit(torque)
                self.idling = True

                self.hex_write_ctrl([0, -0.5, 0.5] * 6)
                self.Ahrs.reset_yaw()
                self.Ahrs.reset_relative_position()
                self.dynamic_step_ctr = 0
                print_sometimes("Idling", 0.1)
                time.sleep(0.1)
            else:
                if self.idling:
                    # Make legs soft
                    torque = dict(zip(self.ids, itertools.repeat(self.max_servo_torque)))
                    self.dxl_io.set_max_torque(torque)
                    self.dxl_io.set_torque_limit(torque)
                    self.idling = False
                    print("Active")

                # Read robot servos and hardware and turn into observation for nn
                clipped_turn = -turn

                if abs(clipped_turn) > 0.47:
                    #self.Ahrs.reset_yaw()

                    speed = dict(zip(self.ids, itertools.repeat(int(self.max_servo_speed * np.maximum(vel, 0.3)))))
                    self.dxl_io.set_moving_speed(speed)
                    self.dyn_speed = vel

                    policy_obs = self.hex_get_obs_turn(clipped_turn)
                    if clipped_turn > 0:
                        policy_act, _ = self.nn_policy_cw.predict(policy_obs, deterministic=True)
                    else:
                        policy_act, _ = self.nn_policy_ccw.predict(policy_obs, deterministic=True)
                    self.hex_write_ctrl_nn(policy_act, mode="turn")
                else:
                    if self.dyn_speed < 0.95:
                        self.dyn_speed = 1.0
                        speed = dict(zip(self.ids, itertools.repeat(int(self.max_servo_speed))))
                        self.dxl_io.set_moving_speed(speed)

                    if button_x:
                        speed = dict(zip(self.ids, itertools.repeat(int(self.max_servo_speed * vel))))
                        self.dxl_io.set_moving_speed(speed)

                        policy_obs = self.hex_get_obs_direct(clipped_turn)
                        policy_act, _ = self.nn_policy_direct.predict(policy_obs, deterministic=True)
                        self.hex_write_ctrl_nn(policy_act, mode="direct")
                        self.dynamic_step_ctr = np.minimum(self.dynamic_step_ctr + 1, self.config["dynamic_max_steps"])
                    else:
                        target_angles = self.calc_target_angles(clipped_turn)
                        self.hex_write_ctrl(target_angles)

            while time.time() - iteration_starttime < self.config["update_period"]: pass

            # TMP DEBUG
            if button_x_event:
                self.Ahrs.reset_yaw()
                self.Ahrs.reset_relative_position()
            self.Ahrs.update()
            pos_rob_relative, vel_rob_relative = self.Ahrs.get_relative_position_and_velocity()
            xd, yd, zd = vel_rob_relative

            print(self.Ahrs.position_rob, pos_rob_relative)

    def hex_write_ctrl(self, joint_angles):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        joint_angles_normed = np.clip(np.array(joint_angles) / 2.618, -1, 1) # [-1,1] corresponding to bounds on servos
        joint_angles_servo = (joint_angles_normed * 0.5 + 0.5) * 1023

        scaled_act = np.array([np.asscalar(joint_angles_servo[i]) for i in range(18)]).astype(np.uint16)

        # Reverse servo signs for right hand servos (This part is retarded and will need to be fixed)
        scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1] = 1023 - scaled_act[np.array([4,5,6,10,11,12,16,17,18])-1]

        scrambled_ids = list(range(1, 19))
        np.random.shuffle(scrambled_ids)
        scrambled_acts = [scaled_act[si - 1] for si in scrambled_ids]
        scaled_act_dict = dict(zip(scrambled_ids, scrambled_acts))

        if self.config["motors_on"]:
            self.dxl_io.set_goal_position(scaled_act_dict)

    def hex_write_ctrl_nn(self, nn_act, mode="direct"):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        nn_act_clipped = np.tanh(nn_act)
        self.prev_act = nn_act_clipped

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = None
        if mode == "direct":
            scaled_act = np.array(
                [(np.asscalar(nn_act_clipped[i]) * 0.5 + 0.5) * self.direct_joints_10bit_diff[i] + self.direct_joints_10bit_low[i] for i
                 in range(18)]).astype(np.uint16)
        if mode == "turn":
            scaled_act = np.array(
                [(np.asscalar(nn_act_clipped[i]) * 0.5 + 0.5) * self.turn_joints_10bit_diff[i] + self.turn_joints_10bit_low[i] for
                 i
                 in range(18)]).astype(np.uint16)
        assert scaled_act is not None

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

    def calc_target_angles(self, turn):
        contacts = self.read_contacts()

        x_mult_arr = [np.minimum(self.x_mult + turn * self.config["turn_coeff"], 0.08), np.minimum(self.x_mult - turn * self.config["turn_coeff"], 0.08)] * 3

        targets = []
        for i in range(6):
            x_cyc = np.sin(self.angle * 2 * np.pi + self.phases[i])
            z_cyc = np.cos(self.angle * 2 * np.pi + self.phases[i])

            target_x = x_cyc * x_mult_arr[i]
            target_y = self.y_offset

            if x_cyc < 0 and z_cyc > 0.0:
                self.dyn_z_lb_array[i] = self.z_lb

            if contacts[i] < 0:
                self.dyn_z_lb_array[i] = z_cyc
                self.poc_array[i] = 1
            else:
                if self.poc_array[i] == 1:
                    self.poc_array[i] = z_cyc
                self.dyn_z_lb_array[i] = self.poc_array[i] - self.config["z_pressure_coeff"]

            target_z = np.maximum(z_cyc, self.dyn_z_lb_array[i]) * self.z_mult + self.z_offset
            targets.append([target_x, target_y, target_z])

        joint_angles = self.my_ikt(targets, self.y_offset)
        self.angle += self.angle_increment

        return joint_angles

    def hex_get_obs_turn(self, heading_spoof_angle=0):
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

        # Turn servo positions into [-1,1] for nn
        joints_normed = ((servo_positions - self.turn_joints_10bit_low) / self.turn_joints_10bit_diff) * 2 - 1
        obs = np.concatenate((quat, vel_rob, [yaw], [0], joints_normed))

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

        pos_rob_relative, vel_rob_relative = self.Ahrs.get_relative_position_and_velocity()
        xd, yd, zd = vel_rob_relative

        # Avg vel
        self.xd_queue.append(xd)
        if len(self.xd_queue) > 15:
            del self.xd_queue[0]
        avg_vel = sum(self.xd_queue) / len(self.xd_queue)
        avg_vel = avg_vel / 0.15 - 1

        # Turn servo positions into [-1,1] for nn
        joints_normed = ((servo_positions - self.direct_joints_10bit_low) / self.direct_joints_10bit_diff) * 2 - 1

        # Torques
        if self.config["velocities_and_torques"]:
            torques_normed = self.get_normalized_torques()
            joint_torques = torques_normed * 1.5
        else:
            joint_torques = [0] * 18

        joint_velocities = [0] * 18

        self.dynamic_time_feature = (float(self.dynamic_step_ctr) / self.config["dynamic_max_steps"]) * 2 - 1

        if not self.config["velocities_and_torques"]:
            # torso_quat, torso_vel, torso_pos, [signed_deviation], time_feature, [avg_vel], scaled_joint_angles, self.prev_act
            obs = np.concatenate((quat, vel_rob_relative, pos_rob_relative, [yaw], [self.dynamic_time_feature], [avg_vel], joints_normed, self.prev_act))
        else:
            obs = np.concatenate((quat, vel_rob_relative, pos_rob_relative, [yaw], [self.dynamic_time_feature], [avg_vel], joints_normed, joint_torques, joint_velocities))

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

    def my_ikt(self, target_positions, rotation_overlay=None):
        # raise NotImplementedError
        rotation_angles = np.array([np.pi / 4, np.pi / 4, 0, 0, -np.pi / 4, -np.pi / 4])
        if rotation_overlay is not None:
            rotation_angles += rotation_overlay
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], tp[1])
            joint_angles.extend(self.single_leg_ikt(tp_rotated))
        return joint_angles

    def my_ikt_robust(self, target_positions, rotation_overlay=None):
        # raise NotImplementedError
        def find_nearest_valid_point(xyz_query, rot_angle=0):
            sol = self.single_leg_ikt(xyz_query)
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
                sol = self.single_leg_ikt(cur_xyz_query)
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
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], tp[1])
            joint_angles.extend(find_nearest_valid_point(tp_rotated, rotation_angles[i]))
        return joint_angles

    def rotate_eef_pos(self, eef_xyz, angle, y_offset):
        return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

    def single_leg_ikt(self, eef_xyz):
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

