import logging
import sys

import Adafruit_PCA9685

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time
import random
import numpy as np
import logging
import pyrealsense2 as rs
import pygame
import os
import yaml
import math as m
import quaternion
from scripts.agent.trajectoryfollower import TrajectoryFollower
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import datetime


datatypes = ["position_rs","velocity_rs","acceleration_rs","rotation_rob","angular_velocity_rob",
             "angular_acceleration_rob","tracker_confidence","mapper_confidence","position_rob","velocity_rob",
             "acceleration_rob","euler_rob","velocity_glob","acceleration_glob","timestamp","frame"]


class JoyController:
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
        turn, throttle = [self.joystick.get_axis(3), self.joystick.get_axis(1)]
        button_A = self.joystick.get_button(0)
        button_B = self.joystick.get_button(1)
        pygame.event.clear()
        turn = -turn # [-.1, .1]
        throttle = np.clip(throttle * -1, 0, 1)  # [0, 1]
        return throttle, turn, button_A, button_B


class AHRS_RS:
    def __init__(self):
        print("Initializing the rs_t265. ")
        self.rs_to_world_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        # added new
        device = self.cfg.resolve(self.pipe).get_device()
        pose_sensor = device.first_pose_sensor()
        #pose_sensor.set_option(rs.option.enable_map_relocalization, 0)
        pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
        #pose_sensor.set_option(rs.option.enable_motion_correction, 0)
        pose_sensor.set_option(rs.option.enable_relocalization, 0)
        # RS2_OPTION_ENABLE_POSE_JUMPING
        # RS2_OPTION_ENABLE_MAPPING
        # RS2_OPTION_ENABLE_RELOCALIZATION
        # RS2_OPTION_ENABLE_MAP_PRESERVATION

        self.pipe.start(self.cfg)
        self.rs_frame = None
        print("Finished initializing the rs_t265. ")

    def rs_cb(self, data_frame):
        with self.rs_lock:
            self.rs_frame = data_frame

    def _quat_to_euler(self, w, x, y, z):
        pitch =  -m.asin(2.0 * (x*z - w*y))
        roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return (roll, pitch, yaw)

    def update(self):
        while True:
            frames = self.pipe.wait_for_frames()
            pose = frames.get_pose_frame()
            if pose:
                data = pose.get_pose_data()
                data_dict = {}

                # Metadata
                data_dict["timestamp"] = frames.get_timestamp()
                #frame = frames.get_pose_frame() # !!LOL THIS MAKES THE T265 CRASH!!
                data_dict["frame"] = frames.frame_number
                
                # Raw data (rotation axes are permuted according how the RS axes are oriented wrt world axes)
                data_dict["position_rs"] = np.array([data.translation.x, data.translation.y, data.translation.z])
                data_dict["velocity_rs"] = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
                data_dict["acceleration_rs"] = np.array([data.acceleration.x, data.acceleration.y, data.acceleration.z])
                data_dict["rotation_rob"] = np.array([data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y])
                data_dict["angular_velocity_rob"] = np.array([data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y])
                data_dict["angular_acceleration_rob"] = np.array([data.angular_acceleration.z, data.angular_acceleration.x, data.angular_acceleration.y])
                data_dict["tracker_confidence"] = np.array(data.tracker_confidence)
                data_dict["mapper_confidence"] = np.array(data.mapper_confidence)

                # Corrected to robot frame
                data_dict["position_rob"] = self.rs_to_world_mat @ data_dict["position_rs"]
                data_dict["velocity_rob"] = self.rs_to_world_mat @ data_dict["velocity_rs"]
                data_dict["acceleration_rob"] = self.rs_to_world_mat @ data_dict["acceleration_rs"]
                data_dict["euler_rob"] = self._quat_to_euler(*data_dict["rotation_rob"])

                # Correct vel_rob to map frame using orientation quat:
                rotation_rob_matrix = quaternion.as_rotation_matrix(np.quaternion(*data_dict["rotation_rob"]))
                data_dict["velocity_glob"] = np.matmul(rotation_rob_matrix.T, data_dict["velocity_rob"])
                data_dict["acceleration_glob"] = np.matmul(rotation_rob_matrix.T, data_dict["acceleration_rob"])

                # Changes (get outputs from dictinonary keys using the following changes):
                # vel_rob -> velocity_glob
                # angular_vel_rob -> angular_velocity_rob
                return data_dict


class PWMDriver:
    def __init__(self, motors_on, pwm_freq):
        self.motors_on = motors_on
        if not self.motors_on:
            print("Motors OFF, not initializing the PWMdriver. ")
            return
        self.pwm_freq = pwm_freq
        self.servo_ids = [0, 1] # MOTOR IS 0, TURN is 1
        print("Initializing the PWMdriver. ")
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(self.pwm_freq)
        self.arm_escs()
        print("Finished initializing the PWMdriver. ")

    def write_servos(self, vals):
        '''
        :param vals: Throttle commands [0,1] corresponding to min and max values
        :return: None
        '''
        if not self.motors_on:
            print("Motors OFF, writing to servos")
            return

        for id in self.servo_ids:
            pulse_length = ((np.clip(vals[id], 0, 1) + 1) * 1000) / ((1000000. / self.pwm_freq) / 4096.)
            self.pwm.set_pwm(id, 0, int(pulse_length))

    def arm_escs(self):
        if not self.motors_on:
            print("Motors OFF, not arming motors")
            return
        time.sleep(0.1)
        print("Setting escs to lowest value. ")
        self.write_servos([0.50, 0.5])
        time.sleep(0.3)

class SimplexNoise:
    """
    A simplex action noise
    """
    def __init__(self, dim, s1, s2):
        super().__init__()
        self.idx = 0
        self.dim = dim
        self.s1 = s1
        self.s2 = s2
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return np.array([(self.noisefun.noise2d(x=self.idx / self.s1, y=i*10) + self.noisefun.noise2d(x=self.idx / self.s2, y=i*10)) for i in range(self.dim)])

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

class Controller:  
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/default.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.motors_on = self.config["motors_on"]
        print("Initializing the Controller, motors_on: {}".format(self.motors_on))
        self.AHRS = AHRS_RS()
        self.PWMDriver = PWMDriver(self.motors_on, int(1. / self.config["update_period"]))
        self.JOYStick = JoyController()
        self.autonomous = False
        self.opensimple_noisefun = SimplexNoise(2, *self.config["opensimplex_scalars"])
        self.agent = TrajectoryFollower(trajectoryname="lap_r2s4.npy")

    def __enter__(self):
        return self

    def update_AHRS_then_read_state(self):
        """update sensors and return data relevant to the AI agent"""
        data = self.AHRS.update()
        return data["position_rob"], data["velocity_glob"], data["rotation_rob"], data["angular_velocity_rob"]

    def correct_throttle(self, throttle):
        """map throttle from [-1, 1] to [0, 1] interval"""
        return (throttle + 1) / 2

    def get_action(self):
        """
        process input from the joystick. 
        if specified pass control to the AI agent.
        return actions: throttle, turn corresponding to the motors m1 and m2
        """
        throttle, turn, button_A, _ = self.JOYStick.get_joystick_input()
        if self.config["controller_source"] == "nn" and button_A:
            action = self.agent.actonobservation(readstatefunc=self.update_AHRS_then_read_state)
            throttle, turn = self.correct_throttle(action[0]), action[1]
        print(f"throttle: {throttle}. turn: {turn}")
        return throttle, turn

    def action_to_servos(self, action_m_1, action_m_2):
        """map joystick or agent actions to motors"""
        m_1 = np.clip((0.5 * action_m_1 * self.config["motor_scalar"]) + self.config["throttle_offset"], 0.5, 1)
        m_2 = (-action_m_2 / 2) + 0.5
        #centre = 0.58
        #m_2 = (action_m_2 + 1) * centre if action_m_2 < 0 else action_m_2 * (1 - centre) + centre
        return m_1, m_2


    def loop_control(self):
        """
        Target inputs are in radians, throttle is in [0,1]
        Roll, pitch and yaw are in radians. 
        Motor outputs are sent to the motor driver as [0,1]
        """
        print("Starting the control loop")
        while True:
            iteration_starttime = time.time()
            action_m_1, action_m_2 = self.get_action()
            m_1, m_2 = self.action_to_servos(action_m_1, action_m_2) 
            self.PWMDriver.write_servos([m_1, m_2])
            while time.time() - iteration_starttime < self.config["update_period"]: pass

    def gather_data(self):
        data = {k: [] for k in datatypes}
        data["action"] = []
        data["throttleturn"] = []
        print("Starting the control loop")
        try:
            for i in range(self.config["n_data_gathering_steps"]):
                if i % 1000 == 0:
                    print(i)

                iteration_starttime = time.time()
                # Read target control inputs
                throttle, turn, button_A, button_B = self.JOYStick.get_joystick_input()
                # Update sensor data
                rs_dict = self.AHRS.update()

                if button_A:
                    osn = self.opensimple_noisefun()
                    # Generate temporally correclated noise from 0,1 for throttle and -1, 1 for turn
                    m_1 = np.clip(osn[0], -1, 1) * 0.5 + 0.5
                    m_2 = np.clip(osn[1], -1, 1)
                elif button_B:
                    # Generate one sided gaussian correlated noise from 0,1 for throttle and -1, 1 for turn
                    m_1 = np.clip(np.random.randn(1) * 0.5, 0, 1)
                    m_2 = np.clip(np.random.randn(1), -2, 2) * 0.5
                else:
                    # Use joystick inputs
                    m_1, m_2 = throttle, turn

                m_1_scaled, m_2_scaled = self.action_to_servos(m_1, m_2)

                if i % 100 == 0:
                    print("Throttle js: {}, turn js: {}, throttle: {}, turn: {}, button_A: {}, button_B: {} ".format(throttle, turn, m_1_scaled, m_2_scaled, button_A, button_B))

                # Realsense data
                for key in datatypes:
                    data[key].append(rs_dict[key])

                # Action data
                data["action"].append([m_1_scaled, m_2_scaled])
                data["throttleturn"].append([m_1, m_2])

                # Write control to servos
                self.PWMDriver.write_servos([m_1_scaled, m_2_scaled])

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["update_period"]: pass
        except KeyboardInterrupt:
            print("Interrupted by user")

        for _ in range(10):
            self.PWMDriver.write_servos([0, 0.5])

        # Save data
        dir_prefix = os.path.join("data", time.strftime("%Y_%m_%d"), 'run'.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3)))
        if not os.path.exists(dir_prefix):
            os.makedirs(dir_prefix)
        prefix = 'buggy_'

        for key in data.keys():
            np.save(os.path.join(dir_prefix, prefix + key), np.array(data[key], dtype=np.float32))

        # Name changes:
        # position -> position_rob
        # vel -> velocity_glob
        # rotation -> rotation_rob
        # angular -> angular_velocity_rob

        print("Saved data")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        before exit save episode history, turn the wheels to the side so buggy 
        dont run straight into the wall and set minimal throttle
        """
        self.agent.saveepisode(etype='real')
        self.PWMDriver.write_servos([0.5, 0])

if __name__ == "__main__":
    with Controller() as controller:
        #controller.gather_data()
        controller.loop_control()

