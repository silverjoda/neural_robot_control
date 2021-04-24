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
import os
import yaml
import math as m
import quaternion
import random



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
        acc_x = self._read_raw_data(
            AHRS.ACCEL_XOUT_H) / AHRS.ACC_SENSITIVITY
        acc_y = self._read_raw_data(
            AHRS.ACCEL_YOUT_H) / AHRS.ACC_SENSITIVITY
        acc_z = self._read_raw_data(
            AHRS.ACCEL_ZOUT_H) / AHRS.ACC_SENSITIVITY

        return acc_x, acc_y, acc_z 


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
        self.timestamp = time.time()
        print("Finished initializing the rs_t265. ")

    def rs_cb(self, data_frame):
        with self.rs_lock:
            self.rs_frame = data_frame

    def _quat_to_euler(self, w, x, y, z):
        pitch =  -m.asin(2.0 * (x*z - w*y))
        roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return (roll, pitch, yaw)
    
    def wait_until_first_rs_frame(self):
        while self.rs_frame is None: pass

    def update(self):
        self.timestamp = time.time()

        if self.rs_frame is not None:
            with self.rs_lock:
                data = self.rs_frame.as_pose_frame().get_pose_data()

            position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
            vel_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            acc_rs = np.array([data.acceleration.x, data.acceleration.y, data.acceleration.z])
            position_rob = self.rs_to_world_mat @ position_rs
            vel_rob = self.rs_to_world_mat @ vel_rs
            acc_rob = self.rs_to_world_mat @ acc_rs

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
            acc_rob = [0., 0., 0.]
            print("RS FRAME WAS NONE, RETURNING DEFAULTS")

        return acc_rob



if __name__=="__main__":
    print("Initializing sensors")
    ahrs_imu = AHRS()
    ahrs_rs = AHRS_RS()
    print("Initialized sensors, sleeping 1s")

    print("Waiting for first rs frame...")
    ahrs_rs.wait_until_first_rs_frame()

    n_samples = 200
    sample_period = 0.01
    print(f"Starting {n_samples} samples of capture") 
    ahrs_imu_data_list = []
    ahrs_rs_data_list = []

    for _ in range(n_samples):
        t1 = time.time()
        imu_x, imu_y, imu_z = ahrs_imu.update()
        rs_x, rs_y, rs_z = ahrs_rs.update()
        ahrs_imu_data_list.append(imu_x)
        ahrs_rs_data_list.append(rs_x)

        while time.time() - t1 < sample_period: pass 

    import matplotlib.pyplot as plt
    t = range(len(ahrs_imu_data_list))
    print(f"Showing {len(t)} data points")
    plt.plot(t, ahrs_imu_data_list, 'b') 
    plt.plot(t, ahrs_rs_data_list, 'r') 
    plt.show()

