## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import math as m
import quaternion
import numpy as np
try:
    pipeline_t265 = rs.pipeline()
    config_t265 = rs.config()
    config_t265.enable_stream(rs.stream.pose)
    pipeline_t265.start(config_t265)
    
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)

    # Start streaming
    pipeline.start(config)

    while True:
        frames = pipeline_t265.wait_for_frames()

        # Fetch pose frame
        pose = frames.get_pose_frame()
        if pose:
            # Print some of the pose data to the terminal
            data = pose.get_pose_data()

            # Euler angles from pose quaternion
            # See also https://github.com/IntelRealSense/librealsense/issues/5178#issuecomment-549795232
            # and https://github.com/IntelRealSense/librealsense/issues/5178#issuecomment-550217609

            w = data.rotation.w
            x = data.rotation.z
            y = data.rotation.x
            z = data.rotation.y

            pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi
            roll = (
                m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
                * 180.0
                / m.pi
            )
            yaw = (
                m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
                * 180.0
                / m.pi
            )

            print("Frame #{}".format(pose.frame_number))
            print(
                "RPY [deg]: Roll: {0:.7f}, Pitch: {1:.7f}, Yaw: {2:.7f}".format(
                    roll, pitch, yaw
                )
            )
                
        quat = quaternion.quaternion(w,x,y,z) 
        rot_mat = quaternion.as_rotation_matrix(quat)

        
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # TODO: THIS IS WRONG
        depth_img_arr = np.asanyarray(depth.get_data())
        img_rot = np.matmul(rot_mat,
                depth_img_arr.reshape(-1)).reshape(*depth_img_arr.shape)
        exit()
        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        coverage = [0]*64
        for y in range(240):
            for x in range(424):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x//10] += 1
            
            if y%20 is 19:
                line = ""
                for c in coverage:
                    line += " .:nhBXWW"[c//25]
                coverage = [0]*64
                print(line)


    exit(0)
except Exception as e:
    print(e)
    pass
