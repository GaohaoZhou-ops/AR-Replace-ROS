#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import os
import rospkg
from sensor_msgs.msg import CompressedImage

# ==============================================================================
# 模板信息类 (Template Information Class)
# ==============================================================================
class Template:
    """
    一个用于存储模板对所有相关信息的类。
    每个实例包含图像、预计算的特征以及自己的一套匹配和验证参数。
    """
    def __init__(self, name, config, default_params, orb_detector):
        self.name = name
        self.template_img_path = self._resolve_ros_path(config['template_path'])
        self.replacement_img_path = self._resolve_ros_path(config['replace_path'])

        user_params = config.get("params", {})
        self.min_match_count = user_params.get("MIN_MATCH_COUNT", default_params["MIN_MATCH_COUNT"])
        self.lowe_ratio_threshold = user_params.get("LOWE_RATIO_THRESHOLD", default_params["LOWE_RATIO_THRESHOLD"])
        self.ransac_reproj_threshold = user_params.get("RANSAC_REPROJ_THRESHOLD", default_params["RANSAC_REPROJ_THRESHOLD"])
        self.min_angle_threshold = user_params.get("MIN_ANGLE_THRESHOLD", default_params["MIN_ANGLE_THRESHOLD"])
        self.max_angle_threshold = user_params.get("MAX_ANGLE_THRESHOLD", default_params["MAX_ANGLE_THRESHOLD"])

        self.template_img = cv2.imread(self.template_img_path, cv2.IMREAD_COLOR)
        self.original_replacement_img = cv2.imread(self.replacement_img_path, cv2.IMREAD_COLOR)
        if self.template_img is None or self.original_replacement_img is None:
            raise IOError(f"Error: Could not load images for template '{name}'.\n"
                          f"Checked paths:\n- {self.template_img_path}\n- {self.replacement_img_path}")

        self.h, self.w, _ = self.template_img.shape
        self.sized_replacement_img = self._create_sized_replacement()
        self.keypoints, self.descriptors = orb_detector.detectAndCompute(self.template_img, None)
        if self.descriptors is None:
            raise ValueError(f"Error: Not enough features found in template '{name}' ({self.template_img_path}).")

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

    def _create_sized_replacement(self):
        h_a, w_a, _ = self.template_img.shape
        h_b, w_b, _ = self.original_replacement_img.shape
        composite_b = np.zeros((h_a, w_a, 3), dtype=np.uint8)
        scale = min(w_a / float(w_b), h_a / float(h_b))
        new_w, new_h = int(w_b * scale), int(h_b * scale)
        resized_b = cv2.resize(self.original_replacement_img, (new_w, new_h))
        x_offset, y_offset = (w_a - new_w) // 2, (h_a - new_h) // 2
        composite_b[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_b
        return composite_b

# ==============================================================================
# ROS 节点类 (ROS Node Class)
# ==============================================================================
class MultiARReplacerNode:
    def __init__(self):
        rospy.init_node('ar_multi_replace_node')
        rospy.loginfo("Starting Multi-Template AR Replacer Node.")

        self._load_config()
        self.orb = cv2.ORB_create(nfeatures=self.orb_feature_count)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.templates = self._setup_templates()
        if not self.templates:
            rospy.logerr("No templates were loaded successfully. Shutting down.")
            rospy.signal_shutdown("Template loading failed.")
            return
        
        # --- 初始化ROS通信 ---
        self.image_pub = rospy.Publisher("/ar_replace/single/compressed", CompressedImage, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed", 
            CompressedImage, 
            self.image_callback, 
            queue_size=1, 
            buff_size=2**24
        )
        rospy.loginfo("Node initialized successfully. Waiting for images...")

    def _load_config(self):
        """从ROS参数服务器加载所有配置"""
        self.orb_feature_count = rospy.get_param('~orb_feature_count', 2000)
        self.default_params = rospy.get_param('~default_params', {})
        self.template_pairs_config = rospy.get_param('~template_pairs', {})
        
        if not self.default_params or not self.template_pairs_config:
            rospy.logfatal("'default_params' or 'template_pairs' not found on Parameter Server. "
                           "Did you load the config YAML file in your launch file?")
            rospy.signal_shutdown("Missing essential parameters.")
    
    def _setup_templates(self):
        """根据加载的配置创建所有Template对象"""
        templates = []
        rospy.loginfo("Loading and preprocessing templates...")
        for name, config in self.template_pairs_config.items():
            try:
                template = Template(name, config, self.default_params, self.orb)
                templates.append(template)
                rospy.loginfo(f"- Successfully loaded template '{name}' "
                              f"(Matches > {template.min_match_count}, "
                              f"Ratio < {template.lowe_ratio_threshold})")
            except (IOError, ValueError, KeyError) as e:
                rospy.logerr(f"Failed to load template '{name}': {e}")
        rospy.loginfo("Template loading complete.")
        return templates

    def get_angle(self, p1, p2, p3):
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0: return 0
        arg = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(arg))

    def is_transformation_valid(self, points, min_angle, max_angle):
        if len(points) != 4 or not cv2.isContourConvex(np.int32(points)):
            return False
        p = [(pt[0][0], pt[0][1]) for pt in points]
        angles = [
            self.get_angle(p[3], p[0], p[1]), self.get_angle(p[0], p[1], p[2]),
            self.get_angle(p[1], p[2], p[3]), self.get_angle(p[2], p[3], p[0])
        ]
        return all(min_angle < angle < max_angle for angle in angles)

    def image_callback(self, ros_data):
        """主回调函数，处理每一帧图像并应用AR"""
        try:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed image: {e}")
            return

        final_frame = frame.copy()
        kp_frame, des_frame = self.orb.detectAndCompute(frame, None)

        if des_frame is None or len(des_frame) == 0:
            self.publish_frame(final_frame, ros_data.header.stamp)
            return

        for tpl in self.templates:
            good_matches = []
            matches = self.bf.knnMatch(tpl.descriptors, des_frame, k=2)
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < tpl.lowe_ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > tpl.min_match_count:
                src_pts = np.float32([tpl.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, tpl.ransac_reproj_threshold)

                if M is not None:
                    pts_corners_tpl = np.float32([[0, 0], [0, tpl.h-1], [tpl.w-1, tpl.h-1], [tpl.w-1, 0]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(pts_corners_tpl, M)

                    if self.is_transformation_valid(dst_corners, tpl.min_angle_threshold, tpl.max_angle_threshold):
                        rospy.loginfo(f"Valid match found for template '{tpl.name}'!")
                        frame_h, frame_w, _ = final_frame.shape
                        warped_replacement = cv2.warpPerspective(tpl.sized_replacement_img, M, (frame_w, frame_h))
                        mask = cv2.warpPerspective(np.ones((tpl.h, tpl.w), dtype=np.uint8) * 255, M, (frame_w, frame_h))
                        mask_inv = cv2.bitwise_not(mask)
                        
                        frame_bg = cv2.bitwise_and(final_frame, final_frame, mask=mask_inv)
                        replacement_fg = cv2.bitwise_and(warped_replacement, warped_replacement, mask=mask)
                        final_frame = cv2.add(frame_bg, replacement_fg)
                        cv2.polylines(final_frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

        self.publish_frame(final_frame, ros_data.header.stamp)

    def publish_frame(self, frame, stamp):
        """编码并发布处理后的帧"""
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
        node = MultiARReplacerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
