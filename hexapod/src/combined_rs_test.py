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
import time
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
    pipe_profile = pipeline.start(config)

    decimate = rs.decimation_filter(8)

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
        #dec_frames = decimate.process(frames).as_frameset()
        depth = frames.get_depth_frame()
        print(type(depth))
        n_depth = 1000
        pc = rs.pointcloud()
        t1 = time.time()
        points = pc.calculate(depth)
        print(time.time() - t1)
        pts_array = np.asarray(points.get_vertices(), dtype=np.ndarray)
        pts_array_decimated = pts_array[np.random.randint(0, len(pts_array), n_depth)]

        pts_numpy = np.zeros((3, len(pts_array_decimated)))

        for i in range(len(pts_array_decimated)):
            pts_numpy[:, i] = pts_array_decimated[i]
 
        img_rot = np.matmul(rot_mat, pts_numpy)


    exit(0)
except Exception as e:
    print(e)
    pass
