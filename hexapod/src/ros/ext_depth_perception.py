#!/usr/bin/env python
import yaml
import rospy
import tf2_ros
import ros_numpy
import time
import threading
import numpy as np
import quaternion

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, Point, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_matrix

class RosCameraInterface:
    def __init__(self, config):
        self.config = config

        rospy.init_node(config["node_name"])
        self.ros_rate = rospy.Rate(config["ros_rate"])

        self.create_publishers()
        self.create_subscribers()

        time.sleep(1.5)

    def create_subscribers(self):
        rospy.Subscriber(self.config["raw_pc_topic"],
                         PointCloud2,
                         self._ros_raw_pc_callback, queue_size=1)

        rospy.Subscriber(self.config["orientation_topic"],
                         Quaternion,
                         self._ros_orientation_callback, queue_size=1)

        self.raw_pc_data = None
        self.orientation_data = None

        self.raw_pc_lock = threading.Lock()
        self.orientation_lock = threading.Lock()

    def create_publishers(self):
        self.transformed_pc_publisher = rospy.Publisher(self.config["transformed_pc_publisher_topic"],
                                                 PointCloud2,
                                                 queue_size=1)
        self.depth_feat_publisher = rospy.Publisher(self.config["depth_feat_publisher_topic"],
                                                        Point,
                                                        queue_size=1)

    def _ros_raw_pc_callback(self, data):
        with self.raw_pc_lock:
            self.raw_pc_data = data

    def _ros_orientation_callback(self, data):
        with self.orientation_lock:
            self.orientation_data = data

    def publish_transformed_pc(self, pc):
        pc_data = np.zeros(len(pc[0]), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('vectors', np.float32, (3,))
        ])

        pc_data['x'] = pc[0, :]
        pc_data['y'] = pc[1, :]
        pc_data['z'] = pc[2, :]
        pc_data['vectors'] = np.arange(len(pc[0]))[:, np.newaxis]

        msg = ros_numpy.msgify(PointCloud2, pc_data)
        msg.header.frame_id = self.config["camera_link"]
        msg.header.stamp = rospy.Time.now()

        self.transformed_pc_publisher.publish(msg)

    def publish_depth_feat(self, feats):
        msg = Point()
        msg.x = feats[0]
        msg.y = feats[1]
        msg.z = feats[2]

        self.depth_feat_publisher.publish(msg)

    def loop_processing(self):
        while not rospy.is_shutdown():
            # Read orientation
            with self.orientation_lock:
                quat = self.orientation_data

            # Read pc and make features
            depth_feats, pc_rot = self.make_depth_features(quat)

            self.publish_transformed_pc(pc_rot)
            self.publish_depth_feat(depth_feats)

            self.ros_rate.sleep()

    def make_depth_features(self, quat):
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        rot_mat = quaternion.as_rotation_matrix(quaternion.quaternion(w, x, y, z))

        # Prepare point cloud
        with self.raw_pc_lock:
            pc = ros_numpy.numpify(self.raw_pc_data).ravel()
        pc_array = np.stack([pc[f] for f in ['x', 'y', 'z']] + [np.ones(pc.size)])
        pc_rot = np.matmul(rot_mat, pc_array)[:3, :]

        ## Calculate depth features
        # Crop pc to appropriate region
        pc_rot = pc_rot[pc_rot[0] < self.config["depth_x_bnd"]]
        pc_rot = pc_rot[np.logical_and(pc_rot[1] < self.config["depth_y_bnd"],
                                       pc_rot[1] > -self.config["depth_y_bnd"])]
        pc_rot = pc_rot[np.logical_and(pc_rot[2]
                                       < self.config["depth_z_bnd_high"], pc_rot[2]
                                       > self.config["depth_z_bnd_low"])]

        # Calculate features
        pc_mean_height = np.mean(pc_rot[2])
        pc_mean_dist = np.mean(pc_rot[0])
        presence = len(pc_rot[0])

        depth_feats = (pc_mean_height, pc_mean_dist, presence)

        return depth_feats, pc_rot

def main():
    with open('configs/default.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cam = RosCameraInterface(config)

if __name__=="__main__":
    main()
