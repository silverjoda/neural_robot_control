#!/usr/bin/env python3
import logging
import math as m
import multiprocessing
import threading
import time
from copy import deepcopy

# Torques are positive upwards and when leg is being pushed backward
import RPi.GPIO as GPIO
import numpy as np
import pybullet as p
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import pygame
import pyrealsense2 as rs
import smbus
import yaml


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
        turn, vel, height = [self.joystick.get_axis(3), self.joystick.get_axis(1), self.joystick.get_axis(4)]
        button_x = self.joystick.get_button(0)
        pygame.event.clear()

        turn = -turn / 2 # [-0.5, 0.5]
        vel = np.maximum(vel * -1, 0)  # [0, 1]
        height = -height

        #print(f"Turn: {turn}, Vel: {vel}, Height: {height}, Button: {button_x}")

        # button_x only when upon press
        if self.button_x_state == 0 and button_x == 1:
            self.button_x_state = 1
            button_x_event = 1
        elif self.button_x_state == 1 and button_x == 0:
            self.button_x_state = 0
            button_x_event = 0
        elif self.button_x_state == 1 and button_x == 1:
            self.button_x_state = 1
            button_x_event = 0
        else:
            self.button_x_state = 0
            button_x_event = 0

        return turn, vel, height, button_x, button_x_event

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

        #self.setup_ros()

        self.rs_to_world_mat = np.array([[0, 0, 1],
                                         [1, 0, 0],
                                         [0, 1, 0]])

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)

        device = self.cfg.resolve(self.pipe).get_device()
        pose_sensor = device.first_pose_sensor()
        pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
        pose_sensor.set_option(rs.option.enable_relocalization, 0)

        self.pipe.start(self.cfg)
        self.timestamp = time.time()

        self.current_heading = 0
        self.yaw_offset = 0
        self.position_offset = np.array([0, 0, 0])
        self.position_rob = np.array([0, 0, 0])
        self.vel_rob = np.array([0, 0, 0])
        self.current_quat = (0, 0, 0, 1) 

        print("Finished initializing the rs_t265. ")

    def update(self, heading_spoof_angle=0):
        self.timestamp = time.time()

        frames = self.pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()

            # Position
            self.position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
            vel_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            self.position_rob = np.matmul(self.rs_to_world_mat, self.position_rs)
            self.vel_rob = np.matmul(self.rs_to_world_mat, vel_rs)

            # Rotation: axes are adjusted according how the RS axes are oriented wrt world axes
            rotation_rs_quat = np.quaternion(data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y)

            angular_vel_rob = (data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y)
            roll, pitch, yaw = self.q2e(rotation_rs_quat.x, rotation_rs_quat.y, rotation_rs_quat.z, rotation_rs_quat.w)

            yaw_corrected = yaw + heading_spoof_angle + self.yaw_offset
            quat_yaw_corrected = self.e2q(roll, pitch, yaw_corrected)
            self.current_quat = quat_yaw_corrected 

        else:
            self.position_rob = np.array([0, 0, 0])
            self.vel_rob = np.array([0, 0, 0])
            rotation_rob_quat = [0, 0, 0, 1]
            angular_vel_rob = [0, 0, 0]
            euler_rob = [0, 0, 0]
            roll, pitch, yaw, yaw_corrected = 0, 0, 0, 0
            quat_yaw_corrected = [0, 0, 0, 1]

        self.current_heading = yaw
        #print(f"Yaw: {yaw}, yaw_corrected: {yaw_corrected}, heading_spoof: {heading_spoof_angle}")

        return roll, pitch, yaw_corrected, quat_yaw_corrected, self.vel_rob, self.timestamp

    def get_depth_feat(self):
        if self.depth_feat_data is None:
            return (0,0,0)
        with self.depth_feat_lock:
            depth_feat = self.depth_feat_data
        return depth_feat

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
        self.yaw_offset = -self.current_heading

    def reset_relative_position(self):
        self.position_offset = -self.position_rob

    def get_relative_position_and_velocity(self):
        # Relative position (in initial frame)
        pos_delta = np.array(self.position_rob) + np.array(self.position_offset)

        # Make yaw correction matrix
        th = self.yaw_offset

        yaw_corr_mat = np.array([[np.cos(th), -np.sin(th), 0],
                                 [np.sin(th), np.cos(th), 0],
                                 [0, 0, 1]])

        pos_delta_corr = np.matmul(yaw_corr_mat, pos_delta)
        vel_corr = np.matmul(yaw_corr_mat, self.vel_rob)

        return pos_delta_corr, vel_corr

    def test(self):
        while True:
            roll, pitch, yaw_corrected, quat_yaw_corrected, xd, _ = self.update()
            print("Roll: {}, Pitch: {}, Yaw: {}, Quat: {}".format(roll, pitch, yaw_corrected, quat_yaw_corrected))
            time.sleep(0.3)

class D435MPIF:
    def __init__(self, config, quat_queue, depth_queue):
        print("Initializing the d435 MP.")

        self.config = config
        self.quat_queue = quat_queue
        self.depth_queue = depth_queue

        self.width = 640
        self.height = 480
        self.format = rs.format.z16
        self.freq = 30

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, self.width, self.height,
                                     self.format, self.freq)

        self.pipeline.start(self.rs_config)
        self.decimate = rs.decimation_filter(8)
        self.current_depth_features = [0, 0, 0]

        self.loop()

    def loop(self):
        while True:
            quat = self.quat_queue.get()
            pc = self._get_depth_pc()
            depth_features = self._get_depth_features(pc, quat)
            self.depth_queue.put(depth_features)

    def _get_depth_pc(self):
        frames = self.pipeline.wait_for_frames()
        dec_frames = self.decimate.process(frames).as_frameset()
        depth = dec_frames.get_depth_frame()

        pc = rs.pointcloud()
        points = pc.calculate(depth)
        pts_array = np.asarray(points.get_vertices(), dtype=np.ndarray)
        pts_np = np.zeros((3, len(pts_array)))

        for i in range(len(pts_array)):
            pts_np[:, i] = pts_array[i]

        pts_np_filt = pts_np[:, np.sum(np.abs(pts_np), axis=0) != 0]

        pts_rearranged = np.empty_like(pts_np_filt)
        pts_rearranged[0,:] = pts_np_filt[2,:]
        pts_rearranged[1,:] = -pts_np_filt[0,:]
        pts_rearranged[2,:] = -pts_np_filt[1,:]

        return pts_rearranged
    
    def _get_depth_features(self, pc, quat):
        if quat is None:
            return (0, 0, 0), None

        if pc is None:
            return (0, 0, 0), None

        euler_x, euler_y, euler_z = p.getEulerFromQuaternion(quat)
        rot_quat_zero_yaw = p.getQuaternionFromEuler([euler_x, euler_y, 0])
        rot_mat_zero_yaw = np.array(p.getMatrixFromQuaternion(rot_quat_zero_yaw)).reshape(3,3)

        pc_rot = np.matmul(rot_mat_zero_yaw[:3, :3], pc)

        ## Calculate depth features
        # Crop pc to appropriate region
        pc_rot = pc_rot[:, pc_rot[0] < self.config["depth_x_bnd"]]
        pc_rot = pc_rot[:, np.logical_and(pc_rot[1] < self.config["depth_y_bnd"],
                                          pc_rot[1] > -self.config["depth_y_bnd"])]
        pc_rot = pc_rot[:, np.logical_and(pc_rot[2]
                                          < self.config["depth_z_bnd_high"], pc_rot[2]
                                          > self.config["depth_z_bnd_low"])]

        height_correction = np.sin(euler_y) * 0.3

        # Calculate features
        pc_mean_height = np.mean(pc_rot[2] + height_correction)
        pc_mean_dist = np.mean(pc_rot[0])
        presence = len(pc_rot[0])

        return pc_mean_height, pc_mean_dist, presence

class D435CameraMP:
    def __init__(self, config):
        print("Initializing the d435.")

        self.config = config

        self.quat_queue = multiprocessing.Queue()
        self.depth_queue = multiprocessing.Queue()
        
        self.p = multiprocessing.Process(target=D435MPIF, args=(config, self.quat_queue, self.depth_queue))
        self.p.start()

        self.current_depth_feats = [0, 0, 0]
        self.quat_queue.put([0, 0, 0, 1])

    def update_orientation(self, quat):
        if self.quat_queue.empty():
            self.quat_queue.put(quat)
        if not self.depth_queue.empty():
            self.current_depth_feats = self.depth_queue.get()
        return self.current_depth_feats

class D435CameraT:
    def __init__(self, config):
        print("Initializing the d435, threaded version.")

        self.config = config

        self.width = 640
        self.height = 480
        self.format = rs.format.z16
        self.freq = 30

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, self.width, self.height,
                                     self.format, self.freq)
        self.pipeline.start(self.rs_config)
        self.decimate = rs.decimation_filter(8)

        self.current_depth_features = [0, 0, 0]
        self.current_orientation = [0, 0, 0, 1]

        self.depth_features_lock = threading.Lock()
        self.orientation_lock = threading.Lock()

        self.loop_thread = threading.Thread(target=self.loop_depth_calculation)
        self.loop_thread.start()

    def loop_depth_calculation(self):
        while True:
            with self.orientation_lock:
                quat = deepcopy(self.current_orientation)
            t1 = time.time()
            pc = self._get_depth_pc()
            depth_features = self._get_depth_features(pc, quat)
            t2 = time.time()
            print(f"Process_time: {t2-t1}")
            with self.depth_features_lock:
                self.current_depth_features = deepcopy(depth_features)

    def get_current_depth_features(self):
        with self.depth_features_lock:
            return self.current_depth_features

    def set_current_orientation(self, quat):
        with self.orientation_lock:
            self.current_orientation = quat

    def _get_depth_pc(self):
        frames = self.pipeline.wait_for_frames()
        dec_frames = self.decimate.process(frames).as_frameset()
        depth = dec_frames.get_depth_frame()

        pc = rs.pointcloud()
        points = pc.calculate(depth)
        pts_array = np.asarray(points.get_vertices(), dtype=np.ndarray)
        pts_np = np.zeros((3, len(pts_array)))

        for i in range(len(pts_array)):
            pts_np[:, i] = pts_array[i]

        pts_np_filt = pts_np[:, np.sum(np.abs(pts_np), axis=0) != 0]

        pts_rearranged = np.empty_like(pts_np_filt)
        pts_rearranged[0,:] = pts_np_filt[2,:]
        pts_rearranged[1,:] = -pts_np_filt[0,:]
        pts_rearranged[2,:] = -pts_np_filt[1,:]

        return pts_rearranged

    def _get_depth_features(self, pc, quat):
        if quat is None:
            return (0,0,0), None

        if pc is None:
            return (0,0,0), None

        euler_x, euler_y, euler_z = p.getEulerFromQuaternion(quat)
        rot_quat_zero_yaw = p.getQuaternionFromEuler([euler_x, euler_y, 0])
        rot_mat_zero_yaw = np.array(p.getMatrixFromQuaternion(rot_quat_zero_yaw)).reshape(3,3)

        pc_rot = np.matmul(rot_mat_zero_yaw[:3, :3], pc)

        ## Calculate depth features

        # Crop pc to appropriate region
        pc_rot = pc_rot[:, pc_rot[0] < self.config["depth_x_bnd"]]
        pc_rot = pc_rot[:, np.logical_and(pc_rot[1] < self.config["depth_y_bnd"],
                                          pc_rot[1] > -self.config["depth_y_bnd"])]
        pc_rot = pc_rot[:, np.logical_and(pc_rot[2]
                                          < self.config["depth_z_bnd_high"], pc_rot[2]
                                          > self.config["depth_z_bnd_low"])]

        height_correction = np.sin(euler_y) * 0.3

        # Calculate features
        pc_mean_height = np.mean(pc_rot[2] + height_correction)
        pc_mean_dist = np.mean(pc_rot[0])
        presence = len(pc_rot[0])

        return pc_mean_height, pc_mean_dist, presence

def q2e(x, y, z, w):
    pitch = -m.asin(2.0 * (x * z - w * y))
    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return (roll, pitch, yaw)

def e2q(roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return (qx, qy, qz, qw)

def read_contacts(leg_sensor_gpio_inputs):
    return [GPIO.input(ipt) * 2 - 1 for ipt in leg_sensor_gpio_inputs]

def test_d435_threaded():
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    depth_cam = D435CameraT(config)
    
    while True:
        t1 = time.time()
        quat = [0,0,0,1]
        depth_cam.set_current_orientation(quat)
        d_feat = depth_cam.get_current_depth_features()
        t2 = time.time()
        #print(f"Time taken for iteration: {t2-t1}")
        print(d_feat)
        time.sleep(0.3)

def test_d435_mp():
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    depth_cam = D435CameraMP(config)
    
    while True:
        t1 = time.time()
        quat = [0,0,0,1]
        d_feat = depth_cam.update_orientation(quat)
        t2 = time.time()
        print(f"Time taken for iteration: {t2-t1}")
        print(d_feat)
        time.sleep(0.0)

def test_ahrs_rs():
    ahrs = AHRS_RS()

    print("Starting ahrs test")
    while True:
        data = ahrs.update(0)
        print(data)
        time.sleep(0.2)

def main():
    #test_d435_threaded()
    test_d435_mp()
    #test_ahrs_rs()

if __name__=="__main__":
    main()

    

