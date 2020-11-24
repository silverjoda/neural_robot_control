#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#####################################################
##           librealsense T265 example             ##
#####################################################

import time

# First import the library
import pyrealsense2 as rs

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()
cfg.enable_stream(rs.stream.pose)

# Start streaming with requested config
pipe.start(cfg)


try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()

        # Fetch pose frame
        pose = frames.get_pose_frame()
        
        if pose:
            # Print some of the pose data to the terminal
            data = pose.get_pose_data()
            print("Frame #{}".format(pose.frame_number))
            print("Position: {}".format(data.translation))
            print("Rotation: {}".format(data.rotation))
            print("Velocity: {}".format(data.velocity))
            print("Angular velocity: {}".format(data.angular_velocity))
            print("Acceleration: {}\n".format(data.acceleration))
            print("Angular acceleration: {}\n".format(data.angular_acceleration))
            print("Tracker confidence: {}\n".format(data.tracker_confidence))
            print("Mapper confidence: {}\n".format(data.mapper_confidence))
            time.sleep(0.05)
            

finally:
    pipe.stop()
