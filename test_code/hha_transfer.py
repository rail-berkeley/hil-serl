#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import time

class HHAEncoder:
    def __init__(self):
        rospy.init_node('hha_encoder_node')
        self.bridge = CvBridge()
        self.intrinsics = None

        rospy.Subscriber('/camera_l/depth/image_raw', Image, self.depth_callback, queue_size=1)
        rospy.Subscriber('/camera_l/depth/camera_info', CameraInfo, self.camera_info_callback)

        self.hha_pub = rospy.Publisher('/camera/depth/hha_image', Image, queue_size=1)
        rospy.loginfo("HHA Encoder Node Initialized.")

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.intrinsics = (self.fx, self.fy, self.cx, self.cy)
        rospy.loginfo_once("Camera intrinsics received.")

    def depth_callback(self, msg):
        if self.intrinsics is None:
            return
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32) / 1000.0  # Convert mm to meters if needed
            start = time.perf_counter()
            hha_img = self.compute_HHA(depth_img, self.intrinsics)
            end = time.perf_counter()
            print(f"hha consume time: {end - start:.6f} sec")
            hha_msg = self.bridge.cv2_to_imgmsg(hha_img, encoding="bgr8")
            hha_msg.header = msg.header
            self.hha_pub.publish(hha_msg)
        except Exception as e:
            rospy.logerr(f"Failed to convert depth image: {e}")

    def compute_HHA(self, depth, intrinsics):
        fx, fy, cx, cy = intrinsics
        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        disparity = fx / (depth + 1e-6)

        x = (xx - cx) * depth / fx
        y = (yy - cy) * depth / fy
        z = depth
        mask = (depth > 0)
        points = np.stack((x, y, z), axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[mask].reshape(-1, 3))

        if len(pcd.points) < 100:
            return np.zeros((H, W, 3), dtype=np.uint8)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        normals = np.zeros((H, W, 3))
        normals_flat = np.asarray(pcd.normals)
        normals[mask] = normals_flat

        gravity = np.array([0, -1, 0])
        dot = np.clip(np.sum(normals * gravity, axis=-1), -1.0, 1.0)
        angle = np.arccos(dot) * (180.0 / np.pi)
        height = y

        hha = np.stack((disparity, height, angle), axis=-1)
        hha -= hha.min(axis=(0, 1), keepdims=True)
        hha /= (hha.max(axis=(0, 1), keepdims=True) + 1e-6)
        hha *= 255.0
        return hha.astype(np.uint8)

if __name__ == '__main__':
    try:
        HHAEncoder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

