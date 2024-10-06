import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d as tf3d
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_srvs.srv import Empty
from sensor_msgs_py import point_cloud2  # 用于解析 PointCloud2
import apriltag  # AprilTag库
import pickle
from .KFClass import KalmanFilterWrapper
from .EKFClass import ExtendedKalmanFilterClass


class AprilTagNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("apriltag_node")

        # Declare and read parameters
        self.declare_parameter("marker_size", 0.0625)
        self.declare_parameter("image_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "")

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.get_logger().info(f"Marker size: {self.marker_size}")
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.get_logger().info(f"Image topic: {image_topic}")
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.get_logger().info(f"Image info topic: {info_topic}")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.get_logger().info(f"camera frame: {self.camera_frame}")

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "apriltag_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "apriltag_markers", 10)
        self.tip_pub = self.create_publisher(PointStamped, "tool_tip_position", 10)
        self.tool_marker_pub = self.create_publisher(PointStamped, "tool_marker_position", 10)
        self.apriltag_image_pub = self.create_publisher(Image, "apriltag_image", 10)

        # Initialize AprilTag detector
        self.tag_detector = apriltag.Detector()
        self.bridge = CvBridge()

        # 相机内参
        self.fx = 641.9315185546875
        self.fy = 641.9315185546875
        self.cx = 643.0005493164062
        self.cy = 362.68548583984375

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        self.destroy_subscription(self.info_sub)  # 假设相机参数不会改变

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        # Convert ROS image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        # 检测AprilTag
        tags = self.tag_detector.detect(cv_image)

        # 初始化AprilTag消息
        markers = ArucoMarkers()  # 使用自定义消息类型ArucoMarkers
        pose_array = PoseArray()
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        for tag in tags:
            # 估算位姿（旋转向量和位移向量）
            rvec, tvec = self.estimate_pose(tag.corners)

            # 创建Pose消息
            pose = Pose()
            pose.position.x = tvec[0]
            pose.position.y = tvec[1]
            pose.position.z = tvec[2]
            rot_matrix = cv2.Rodrigues(rvec)[0]
            quat = tf3d.quaternions.mat2quat(rot_matrix)
            pose.orientation.x = quat[1]
            pose.orientation.y = quat[2]
            pose.orientation.z = quat[3]
            pose.orientation.w = quat[0]

            pose_array.poses.append(pose)
            markers.marker_ids.append(tag.tag_id)

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers)

        # 绘制检测到的AprilTag
        for tag in tags:
            for i in range(4):
                pt1 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                pt2 = (int(tag.corners[(i + 1) % 4][0]), int(tag.corners[(i + 1) % 4][1]))
                cv2.line(cv_image, pt1, pt2, (0, 255, 0), 2)

        # 显示结果图像
        cv2.imshow("AprilTag Detection", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        image_message = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
        self.apriltag_image_pub.publish(image_message)

    def estimate_pose(self, corners):
        """根据AprilTag的角点估算姿态"""
        object_points = np.array([[0, 0, 0],
                                  [self.marker_size, 0, 0],
                                  [self.marker_size, self.marker_size, 0],
                                  [0, self.marker_size, 0]], dtype=np.float32)
        image_points = np.array(corners, dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.intrinsic_mat, self.distortion)
        return rvec, tvec


def main():
    rclpy.init()
    node = AprilTagNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
