## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np

width = 424
height = 240

try:
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    
    #Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 6)

    # Start streaming
    pipeline.start(config)

    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # Get numpy array from depth object
        depth_img_arr = np.asanyarray(depth.get_data())

        print(depth_img_arr.max())
        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        if True:
            coverage = [0]*64
            for y in range(height):
                for x in range(width):
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
    print("ERROR", e)
    pass
