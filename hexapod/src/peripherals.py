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
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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
        self.current_heading = 0

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        self.pipe.start(self.cfg, callback=self.rs_cb)
        self.timestamp = time.time()
        self.rs_lock = threading.Lock()

        self.rs_frame = None
        self.yaw_offset = 0

        print("Finished initializing the rs_t265. ")

    def rs_cb(self, data_frame):
        with self.rs_lock:
            self.rs_frame = data_frame

    def update(self, heading_spoof_angle=0):
        heading_spoof_angle -= self.yaw_offset
        self.timestamp = time.time()

        if self.rs_frame is not None:
            with self.rs_lock:
                data = self.rs_frame.as_pose_frame().get_pose_data()

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
        #print(f"Yaw: {yaw}, yaw_corrected: {yaw_corrected}, heading_spoof: {heading_spoof_angle}")

        return roll, pitch, yaw_corrected, quat_yaw_corrected, vel_rob, self.timestamp

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
