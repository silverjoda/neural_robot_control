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
from scripts.agent.agent import Agent
from scripts.agent.trajectory import Trajectory2d
from scripts.datamanagement.configsmanager import ConfigsManager
cm = ConfigsManager()
import scripts.datamanagement.datamanager as dm
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import datetime


def replace_symbols(var: str, to_replace: str, replacement: str):
    """replace all required symbols in the string with desired"""
    for symbol in to_replace:
        var = var.replace(symbol, replacement)
    return var


def datetime_to_str():
    """date and time to a string"""
    s = str(datetime.datetime.today())
    return replace_symbols(s, to_replace=' -:.', replacement='_')


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

                # Metadata
                timestamp = frames.get_timestamp()
                #frame = frames.get_pose_frame() # !!LOL THIS MAKES THE T265 CRASH!!
                frame = frames.frame_number

                
                # Raw data (rotation axes are permuted according how the RS axes are oriented wrt world axes)
                position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
                velocity_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
                acceleration_rs = np.array([data.acceleration.x, data.acceleration.y, data.acceleration.z])
                rotation_rob = np.array([data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y])
                angular_velocity_rob = np.array([data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y])
                angular_acceleration_rob = np.array([data.angular_acceleration.z, data.angular_acceleration.x, data.angular_acceleration.y])
                tracker_confidence = np.array(data.tracker_confidence)
                mapper_confidence = np.array(data.mapper_confidence)

                # Corrected to robot frame
                position_rob = self.rs_to_world_mat @ position_rs
                velocity_rob = self.rs_to_world_mat @ velocity_rs
                acceleration_rob = self.rs_to_world_mat @ acceleration_rs
                euler_rob = self._quat_to_euler(*rotation_rob)

                # Correct vel_rob to map frame using orientation quat:
                rotation_rob_matrix = quaternion.as_rotation_matrix(np.quaternion(*rotation_rob))
                velocity_glob = np.matmul(rotation_rob_matrix.T, velocity_rob)
                acceleration_glob = np.matmul(rotation_rob_matrix.T, acceleration_rob)

                data_dict = {"position_rs" : position_rs,
                             "velocity_rs" : velocity_rs,
                             "acceleration_rs" : acceleration_rs,
                             "rotation_rob" : rotation_rob,
                             "angular_velocity_rob" : angular_velocity_rob,
                             "angular_acceleration_rob" : angular_acceleration_rob,
                             "tracker_confidence" : tracker_confidence,
                             "mapper_confidence" : mapper_confidence,
                             "position_rob" : position_rob,
                             "velocity_rob" : velocity_rob,
                             "acceleration_rob" : acceleration_rob,
                             "euler_rob" : euler_rob,
                             "velocity_glob" : velocity_glob,
                             "acceleration_glob" : acceleration_glob,
                             "timestamp" : timestamp,
                             "frame" : frame}

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
        self.agent = Agent()
        self.trajectory = Trajectory2d(n_waypoints=cm.get(key='env', param='n_waypoints'), 
                                       filename="lap_r3s5.npy")

    def __enter__(self):
        return self

    def update_AHRS_then_read_state(self):
        """update sensors and return data relevant to the AI agent"""
        data = self.AHRS.update()
        pos = data["position_rob"]
        vel = data["velocity_glob"]
        rot = data["rotation_rob"]
        ang = data["angular_velocity_rob"]
        return pos, vel, rot, ang

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
            action = self.agent.observe_then_act(readstatefunc=self.update_AHRS_then_read_state, 
                                                 pathplanfunc=self.trajectory.get_waypoints_vector)
            throttle, turn = self.correct_throttle(action[0]), action[1]
            self.trajectory.update_points_state(self.agent.state.get_pos()[:2])
            print(f"throttle: {throttle}. turn: {turn}. waypoints: {self.trajectory.get_next_unvisited_point()}. pos: {self.agent.state.get_pos()}")
        print(f"throttle: {throttle}. turn: {turn}")
        return throttle, turn

    def action_to_servos(self, action_m_1, action_m_2):
        m_1 = np.clip((0.5 * action_m_1 * self.config["motor_scalar"]) + self.config["throttle_offset"], 0.5, 1)
        #m_2 = (action_m_2 / 2) + 0.5
        center = 0.7
        m_2 = (action_m_2 + 1) * center if action_m_2 < 0 else action_m_2 * (1-center) + center
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

        # Initialize data lists
        data_position_rs = []
        data_velocity_rs = []
        data_acceleration_rs = []
        data_rotation_rob = []
        data_angular_velocity_rob = []
        data_angular_acceleration_rob = []
        data_tracker_confidence = []
        data_mapper_confidence = []
        data_position_rob = []
        data_velocity_rob = []
        data_acceleration_rob = []
        data_euler_rob = []
        data_velocity_glob = []
        data_acceleration_glob = []
        data_timestamp = []
        data_frame = []
        data_action = []
        data_throttleturn = []

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
                data_position_rs.append(rs_dict["position_rs"])
                data_velocity_rs.append(rs_dict["velocity_rs"])
                data_acceleration_rs.append(rs_dict["acceleration_rs"])
                data_rotation_rob.append(rs_dict["rotation_rob"])
                data_angular_velocity_rob.append(rs_dict["angular_velocity_rob"])
                data_angular_acceleration_rob.append(rs_dict["angular_acceleration_rob"])
                data_tracker_confidence.append(rs_dict["tracker_confidence"])
                data_mapper_confidence.append(rs_dict["mapper_confidence"])
                data_position_rob.append(rs_dict["position_rob"])
                data_velocity_rob.append(rs_dict["velocity_rob"])
                data_acceleration_rob.append(rs_dict["acceleration_rob"])
                data_euler_rob.append(rs_dict["euler_rob"])
                data_velocity_glob.append(rs_dict["velocity_glob"])
                data_acceleration_glob.append(rs_dict["acceleration_glob"])
                data_timestamp.append(rs_dict["timestamp"])
                data_frame.append(rs_dict["frame"])

                # Action data
                data_action.append([m_1_scaled, m_2_scaled])
                data_throttleturn.append([m_1, m_2])

                # Write control to servos
                self.PWMDriver.write_servos([m_1_scaled, m_2_scaled])

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["update_period"]: pass
        except KeyboardInterrupt:
            print("Interrupted by user")

        for _ in range(10):
            self.PWMDriver.write_servos([0, 0.5])

        # Save data
        dir_prefix = os.path.join("data", time.strftime("%Y_%m_%d"))
        if not os.path.exists(dir_prefix):
            os.makedirs(dir_prefix)
        prefix = 'buggy_' + ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))

        data_position_rs = np.array(data_position_rs, dtype=np.float32)
        data_velocity_rs = np.array(data_velocity_rs, dtype=np.float32)
        data_acceleration_rs = np.array(data_acceleration_rs, dtype=np.float32)
        data_rotation_rob = np.array(data_rotation_rob, dtype=np.float32)
        data_angular_velocity_rob = np.array(data_angular_velocity_rob, dtype=np.float32)
        data_angular_acceleration_rob = np.array(data_angular_acceleration_rob, dtype=np.float32)
        data_tracker_confidence = np.array(data_tracker_confidence)
        data_mapper_confidence = np.array(data_mapper_confidence)
        data_position_rob = np.array(data_position_rob, dtype=np.float32)
        data_velocity_rob = np.array(data_velocity_rob, dtype=np.float32)
        data_acceleration_rob = np.array(data_acceleration_rob, dtype=np.float32)
        data_euler_rob = np.array(data_euler_rob, dtype=np.float32)
        data_velocity_glob = np.array(data_velocity_glob, dtype=np.float32)
        data_acceleration_glob = np.array(data_acceleration_glob, dtype=np.float32)
        data_timestamp = np.array(data_timestamp)
        data_frame = np.array(data_frame)
        data_action = np.array(data_action)
        data_throttleturn = np.array(data_throttleturn)

        np.save(os.path.join(dir_prefix, prefix + "_position_rs"), data_position_rs)
        np.save(os.path.join(dir_prefix, prefix + "_velocity_rs"), data_velocity_rs)
        np.save(os.path.join(dir_prefix, prefix + "_acceleration_rs"), data_acceleration_rs)
        np.save(os.path.join(dir_prefix, prefix + "_rotation_rob"), data_rotation_rob)
        np.save(os.path.join(dir_prefix, prefix + "_angular_velocity_rob"), data_angular_velocity_rob)
        np.save(os.path.join(dir_prefix, prefix + "_angular_acceleration_rob"), data_angular_acceleration_rob)
        np.save(os.path.join(dir_prefix, prefix + "_tracker_confidence"), data_tracker_confidence)
        np.save(os.path.join(dir_prefix, prefix + "_mapper_confidence"), data_mapper_confidence)
        np.save(os.path.join(dir_prefix, prefix + "_position_rob"), data_position_rob)
        np.save(os.path.join(dir_prefix, prefix + "_velocity_rob"), data_velocity_rob)
        np.save(os.path.join(dir_prefix, prefix + "_acceleration_rob"), data_acceleration_rob)
        np.save(os.path.join(dir_prefix, prefix + "_euler_rob"), data_euler_rob)
        np.save(os.path.join(dir_prefix, prefix + "_velocity_glob"), data_velocity_glob)
        np.save(os.path.join(dir_prefix, prefix + "_acceleration_glob"), data_acceleration_glob)
        np.save(os.path.join(dir_prefix, prefix + "_timestamp"), data_timestamp)
        np.save(os.path.join(dir_prefix, prefix + "_frame"), data_frame)
        np.save(os.path.join(dir_prefix, prefix + "_action"), data_action)
        np.save(os.path.join(dir_prefix, prefix + "_throttleturn"), data_throttleturn)


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
        dm.save_episode(history=self.agent.get_history(), trajectory=self.trajectory.trajectory, 
                        tag=f"realtest_{datetime_to_str()}")
        self.PWMDriver.write_servos([0.5, 0])

if __name__ == "__main__":
    with Controller() as controller:
        #controller.gather_data()
        controller.loop_control()

