import Adafruit_PCA9685
import sys
#import rospy
#import std_msgs
#from geometry_msgs.msg import Pose, PoseStamped, Twist
#from sensor_msgs.msg import Joy
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
import pygame
import os
import yaml
import math as m
import random
from stable_baselines import A2C

# root = logging.getLogger()
# root.setLevel(logging.INFO)
#
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root.addHandler(handler)


class RandomSeq:
    def __init__(self, N, config):
        self.N = N
        self.config = config

    def __getitem__(self, item):
        return np.random.rand(2) * self.config["target_dispersal_distance"] - self.config["target_dispersal_distance"] / 2

    def __len__(self):
        return 2


class WPGenerator:
    def __init__(self, config):
        self.config = config
        self.wp_sequence = None
        if self.config["wp_sequence"] == "two_pole":
            self.wp_sequence = [[2, 0], [-2, 0]]
        if self.config["wp_sequence"] == "four_pole":
            self.wp_sequence = [[2, 0], [0, -2], [-2, 0], [0, 2]]
        if self.config["wp_sequence"] == "rnd":
            self.wp_sequence = RandomSeq(2, config)
        self.wp_idx = 0
        self.N = len(self.wp_sequence)

    def next(self):
        wp = self.wp_sequence[self.wp_idx]
        self.wp_idx = (self.wp_idx + 1) % self.N
        return wp


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
        #print([self.joystick.get_axis(i) for i in range(4)])
        #print(turn, vel)
        button_x = self.joystick.get_button(0)
        pygame.event.clear()

        turn = -turn # [-.5, .5]
        vel = vel * -1  # [0, 1]

        # button_x only when upon press
        #if self.button_x_state == 0 and button_x == 1:
        #    self.button_x_state = 1
        #    button_x = 1
        #elif self.button_x_state == 1 and button_x == 0:
        #    self.button_x_state = 0
        #    button_x = 0
        #elif self.button_x_state == 1 and button_x == 1:
        #    self.button_x_state = 1
        #    button_x = 0
        #else:
        #    self.button_x_state = 0
        #    button_x = 0

        return vel, turn, button_x


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
        self.timestamp = time.time()

        if self.rs_frame is not None:
            with self.rs_lock:
                data = self.rs_frame.as_pose_frame().get_pose_data()

            position_rs = np.array([data.translation.x, data.translation.y, data.translation.z])
            vel_rs = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            position_rob = self.rs_to_world_mat @ position_rs
            vel_rob = self.rs_to_world_mat @ vel_rs

            # Axes are permuted according how the RS axes are oriented wrt world axes
            rotation_rob = (data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y)
            angular_vel_rob = (data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y)
            euler_rob = self._quat_to_euler(*rotation_rob)
        else:
            print("RS frame was None, so returning zero values")
            position_rob = [0., 0., 0.]
            vel_rob =  [0., 0., 0.]
            rotation_rob =  [0., 0., 0., 0.]
            angular_vel_rob =  [0., 0., 0.]
            euler_rob =  [0., 0., 0.]

        return position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, self.timestamp


class PWMDriver:
    def __init__(self, motors_on):
        self.motors_on = motors_on

        if not self.motors_on:
            print("Motors OFF, not initializing the PWMdriver. ")
            return

        self.pwm_freq = 50
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
        self.write_servos([0.55,0.5])
        time.sleep(0.3)


class Controller:
    def __init__(self):
        with open('configs/default.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.motors_on = self.config["motors_on"]

        print("Initializing the Controller, motors_on: {}".format(self.motors_on))

        self.config["obs_dim"], self.config["act_dim"] = 10, 2

        self.AHRS = AHRS_RS()
        self.PWMDriver = PWMDriver(self.motors_on)
        self.JOYStick = JoyController()
        self.policy = self.load_policy(self.config)
        self.waypoint_generator = WPGenerator(self.config)
        self.update_targets()

        print("Finished initializing the Controller")

        self.autonomous = False
        self.fw_dir = True

    def update_targets(self):
        if not hasattr(self, 'target_A'):
            self.target_A = self.waypoint_generator.next()
            self.target_B = self.waypoint_generator.next()
        else:
            self.target_A = self.target_B
            self.target_B = self.waypoint_generator.next()

    def load_policy(self, config):
        logging.info("Loading policy from: \"{}\" ".format(config["policy_path"]))
        policy = A2C.load("agents/{}".format(self.config["policy_path"]))
        return policy

    def get_policy_action(self, obs):
        (throttle, turn ), _ = self.policy.predict(obs)
        return throttle, turn

    def loop_control(self):
        '''
        Target inputs are in radians, throttle is in [0,1]
        Roll, pitch and yaw are in radians. 
        Motor outputs are sent to the motor driver as [0,1]
        '''

        print("Starting the control loop")
        while True:
            iteration_starttime = time.time()

            throttle, turn, autonomous_control = self.JOYStick.get_joystick_input()

            # Update sensor data
            position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, timestamp = self.AHRS.update()
            target_dist = np.sqrt((position_rob[0] - self.target_A[0]) ** 2 + (position_rob[1] - self.target_A[1]) ** 2)
            vel_dirvec = np.sqrt(np.square(vel_rob).sum())

            if target_dist < self.config["target_proximity_threshold"]:
                self.update_targets()

            # Calculate relative positions of targets
            relative_target_A = self.target_A[0] - position_rob[0], self.target_A[1] - position_rob[1]
            relative_target_B = self.target_B[0] - position_rob[0], self.target_B[1] - position_rob[1]

            if self.config["controller_source"] == "nn" and autonomous_control:
                # Make neural network observation vector
                obs = np.concatenate((euler_rob[2:3], vel_rob[0:2], angular_vel_rob[2:3], relative_target_A, relative_target_B)) # Ours
                action_m_1, action_m_2 = self.get_policy_action(obs)
            else:
                action_m_1, action_m_2 = throttle, turn

            if action_m_1 < 0:
                action_m_1 *= 1.8
            m_1, m_2 = np.clip((0.5 * action_m_1 * self.config["motor_scalar"]) + self.config["throttle_offset"], 0, 1) , (action_m_2 / 2) + 0.5

            # Software work-around concerning "double-click" backwards issue
            if self.fw_dir and (vel_dirvec < 0.03) and m_1 < (self.config["throttle_offset"] - 0.03):
                self.fw_dir = False
                self.PWMDriver.write_servos([self.config["throttle_offset"], m_2])
                time.sleep(0.02)
                self.PWMDriver.write_servos([0, m_2])
                time.sleep(0.1)
                self.PWMDriver.write_servos([self.config["throttle_offset"], m_2])
                time.sleep(0.1)

            print(f"Position: {position_rob}, throttle: {throttle}, motor_commands: {m_1}, {m_2}, autonomous: {autonomous_control}, dirvec: {vel_dirvec}, fw_dir: {self.fw_dir}")

            if m_1 > self.config["throttle_offset"] + 0.03:
                self.fw_dir = True

            # Write control to servos
            self.PWMDriver.write_servos([m_1, m_2])

            while time.time() - iteration_starttime < self.config["update_period"]: pass
            
    def gather_data(self, n_iterations=20000):

        # Initialize data lists
        data_position = []
        data_vel = []
        data_rotation = []
        data_angular_vel = []
        data_timestamp = []
        data_action = []

        print("Starting the control loop")
        try:
            for i in range(n_iterations):
                iteration_starttime = time.time()

                # Read target control inputs
                throttle, turn, autonomous_control = self.JOYStick.get_joystick_input()

                # Update sensor data
                position_rob, vel_rob, rotation_rob, angular_vel_rob, euler_rob, timestamp = self.AHRS.update()

                target_dist = np.sqrt(
                    (position_rob[0] - self.target_A[0]) ** 2 + (position_rob[1] - self.target_A[1]) ** 2)

                if target_dist < self.config["target_proximity_threshold"]:
                    self.update_targets()

                # Calculate relative positions of targets
                relative_target_A = self.target_A[0] - position_rob[0], self.target_A[1] - position_rob[1]
                relative_target_B = self.target_B[0] - position_rob[0], self.target_B[1] - position_rob[1]

                if self.config["controller_source"] == "nn" and autonomous_control:
                    # Make neural network observation vector
                    obs = np.concatenate((euler_rob[2:3], vel_rob[0:2], angular_vel_rob[2:3], relative_target_A,
                                          relative_target_B))  # Ours
                    m_1, m_2 = self.get_policy_action(obs)
                else:
                    m_1, m_2 = np.clip((0.5 * throttle * self.config["motor_scalar"]) + self.config["throttle_offset"],
                                       0, 1), (turn / 2) + 0.5

                data_position.append(position_rob)
                data_vel.append(vel_rob)
                data_rotation.append(rotation_rob)
                data_angular_vel.append(angular_vel_rob)
                data_timestamp.append(timestamp)
                data_action.append([throttle, turn])

                # Write control to servos
                #print(m_1, m_2)
                #print(throttle, turn)
                self.PWMDriver.write_servos([m_1, m_2])

                # Publish telemetry values
                # self.ROSInterface.publish_telemetry(timestamp, position_rob, quat_rob)

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["update_period"]: pass
        except KeyboardInterrupt:
            print("Interrupted by user")

        # Save data
        prefix = os.path.join("data", time.strftime("%Y_%m_%d"))
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        prefix = prefix + 'buggy_'.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))

        data_position = np.array(data_position, dtype=np.float32)
        data_vel = np.array(data_vel, dtype=np.float32)
        data_rotation = np.array(data_rotation, dtype=np.float32)
        data_angular_vel = np.array(data_angular_vel, dtype=np.float32)
        data_timestamp = np.array(data_timestamp, dtype=np.float32)
        data_action = np.array(data_action, dtype=np.float32)

        np.save(prefix + "_position", data_position)
        np.save(prefix + "_vel", data_vel)
        np.save(prefix + "_rotation", data_rotation)
        np.save(prefix + "_angular_vel", data_angular_vel)
        np.save(prefix + "_timestamp", data_timestamp)
        np.save(prefix + "_action", data_action)

        print("Saved data")


if __name__ == "__main__":
    controller = Controller()
    controller.loop_control()
