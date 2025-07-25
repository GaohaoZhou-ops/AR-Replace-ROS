#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import os
import rospkg
from sensor_msgs.msg import CompressedImage

class ARReplacerNode:
    def __init__(self):
        """
        ROS节点初始化
        """
        rospy.init_node('ar_replacer_node')
        rospy.loginfo("AR Replacer Node (Single Template) started.")

        # --- 从ROS参数服务器加载所有配置 ---
        self._load_config()

        # --- 初始化CV组件 ---
        self.orb = cv2.ORB_create(nfeatures=self.params['ORB_FEATURE_COUNT'])
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # --- 加载并预处理模板图像 ---
        self.load_templates()
        
        # --- 初始化ROS发布者和订阅者 ---
        self.image_pub = rospy.Publisher("/ar_replace/single/compressed", CompressedImage, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed", 
            CompressedImage, 
            self.image_callback, 
            queue_size=1, 
            buff_size=2**24
        )
        rospy.loginfo("Node initialized successfully. Waiting for images...")

    def _resolve_ros_path(self, path_str):
        """解析ROS风格的路径 e.g., $(find my_pkg)/res/img.png"""
        if path_str.startswith("$(find "):
            parts = path_str.split(")")
            pkg_name = parts[0].replace("$(find ", "")
            relative_path = parts[1].strip("/")
            try:
                rospack = rospkg.RosPack()
                return os.path.join(rospack.get_path(pkg_name), relative_path)
            except rospkg.ResourceNotFound:
                rospy.logerr(f"Package '{pkg_name}' not found for path: {path_str}")
                return None
        return path_str

    def _load_config(self):
        """从ROS参数服务器加载所有配置"""
        try:
            # 使用 '~' 表示私有参数
            self.template_file_path = self._resolve_ros_path(rospy.get_param('~template_path'))
            self.replace_file_path = self._resolve_ros_path(rospy.get_param('~replace_path'))
            self.params = rospy.get_param('~params')
        except KeyError as e:
            rospy.logfatal(f"Parameter {e} not found on server! "
                           "Did you load the config YAML file in your launch file?")
            rospy.signal_shutdown(f"Missing essential parameter: {e}")

    def load_templates(self):
        """加载模板A和替换模板B，并预先计算模板A的特征点。"""
        template_a = cv2.imread(self.template_file_path, cv2.IMREAD_COLOR)
        original_b = cv2.imread(self.replace_file_path, cv2.IMREAD_COLOR)

        if template_a is None or original_b is None:
            rospy.logerr(f"Error: Could not load one or more template images.\n"
                         f"Checked paths:\n- {self.template_file_path}\n- {self.replace_file_path}")
            rospy.signal_shutdown("Template images not found.")
            return
            
        rospy.loginfo("Template images loaded successfully.")
        self.template_a = template_a
        self.h_a, self.w_a, _ = self.template_a.shape
        self.template_b_composite = self._create_composite_b(template_a, original_b)
        self.kp_a, self.des_a = self.orb.detectAndCompute(self.template_a, None)
        
        if self.des_a is None:
            rospy.logerr("Error: Not enough features found in template_a.")
            rospy.signal_shutdown("Failed to compute features for template.")
            
    def _create_composite_b(self, template_a, template_b):
        h_a, w_a, _ = template_a.shape
        h_b, w_b, _ = template_b.shape
        composite_b = np.zeros((h_a, w_a, 3), dtype=np.uint8)
        scale = min(w_a / float(w_b), h_a / float(h_b))
        new_w, new_h = int(w_b * scale), int(h_b * scale)
        resized_b = cv2.resize(template_b, (new_w, new_h))
        x_offset, y_offset = (w_a - new_w) // 2, (h_a - new_h) // 2
        composite_b[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_b
        return composite_b

    def get_angle(self, p1, p2, p3):
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0: return 0
        arg = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(arg))

    def is_transformation_valid(self, points):
        min_angle = self.params['MIN_ANGLE_THRESHOLD']
        max_angle = self.params['MAX_ANGLE_THRESHOLD']
        if len(points) != 4 or not cv2.isContourConvex(np.int32(points)):
            return False
        p = [(pt[0][0], pt[0][1]) for pt in points]
        angles = [
            self.get_angle(p[3], p[0], p[1]), self.get_angle(p[0], p[1], p[2]),
            self.get_angle(p[1], p[2], p[3]), self.get_angle(p[2], p[3], p[0])
        ]
        return all(min_angle < angle < max_angle for angle in angles)

    def image_callback(self, ros_data):
        try:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed image: {e}")
            return

        final_frame = frame.copy()
        kp_frame, des_frame = self.orb.detectAndCompute(frame, None)

        if des_frame is not None and len(des_frame) > 0:
            good_matches = []
            matches = self.bf.knnMatch(self.des_a, des_frame, k=2)
            
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.params['LOWE_RATIO_THRESHOLD'] * n.distance:
                        good_matches.append(m)

            if len(good_matches) > self.params['MIN_MATCH_COUNT']:
                src_pts = np.float32([self.kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.params['RANSAC_REPROJ_THRESHOLD'])

                if M is not None:
                    pts_corners_a = np.float32([[0, 0], [0, self.h_a - 1], [self.w_a - 1, self.h_a - 1], [self.w_a - 1, 0]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(pts_corners_a, M)

                    if self.is_transformation_valid(dst_corners):
                        rospy.loginfo("Valid transformation found! Applying AR replacement.")
                        frame_h, frame_w, _ = frame.shape
                        warped_b = cv2.warpPerspective(self.template_b_composite, M, (frame_w, frame_h))
                        mask = cv2.warpPerspective(np.ones((self.h_a, self.w_a), dtype=np.uint8) * 255, M, (frame_w, frame_h))
                        mask_inv = cv2.bitwise_not(mask)
                        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                        warped_b_fg = cv2.bitwise_and(warped_b, warped_b, mask=mask)
                        final_frame = cv2.add(frame_bg, warped_b_fg)
                        cv2.polylines(final_frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        self.publish_frame(final_frame, ros_data.header.stamp)

    def publish_frame(self, frame, stamp):
        try:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
            self.image_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"Failed to encode and publish image: {e}")

if __name__ == '__main__':
    try:
        node = ARReplacerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass