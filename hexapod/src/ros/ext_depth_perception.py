#!/usr/bin/env python
import yaml
import rospy
import tf2_ros
import time
import threading

from sensor_msgs.msg import PointCloud2
from tf.transformations import euler_from_quaternion, quaternion_matrix

class RosCameraInterface:
    def __init__(self, config):
        self.config = config

        rospy.init_node(config["node_name"])
        self.ros_rate = rospy.Rate(config["ros_rate"])

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.create_publishers()
        self.create_subscribers()

        time.sleep(1.5)

    def create_subscribers(self):
        rospy.Subscriber(self.config["registered_pointcloud_topic"],
                         PointCloud2,
                         self._ros_reg_pc_callback, queue_size=1)

        self.raw_pc_data = None

        self.raw_pc_lock = threading.Lock()
        self.imu_lock = threading.Lock()


    def create_publishers(self):
        pass

def _ros_raw_pc_callback(self, data):
    with self.raw_pc_lock:
        self.raw_pc_data = data

def main():
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cam = RosCameraInterface(config)

if __name__=="__main__":
    main()
