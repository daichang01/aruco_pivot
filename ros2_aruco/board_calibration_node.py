import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d as tf3d
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_srvs.srv import Empty
# from .pivot_calibration import PivotCalibration
from scipy.optimize import least_squares


    
class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Declare and read parameters
        self.declare_parameter("marker_size", 0.0625)
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_250")
        self.declare_parameter("image_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "")

        self.marker_size = (self.get_parameter("marker_size").get_parameter_value().double_value)
        self.get_logger().info(f"Marker size: {self.marker_size}")
        dictionary_id_name = (self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value)
        self.get_logger().info(f"Marker type: {dictionary_id_name}")
        image_topic = (self.get_parameter("image_topic").get_parameter_value().string_value)
        self.get_logger().info(f"Image topic: {image_topic}")
        info_topic = (self.get_parameter("camera_info_topic").get_parameter_value().string_value)
        self.get_logger().info(f"Image info topic: {info_topic}")
        self.camera_frame = (self.get_parameter("camera_frame").get_parameter_value().string_value)
        self.get_logger().info(f"camera frame: {self.camera_frame}")

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.tip_pub = self.create_publisher(PointStamped, "tool_tip_position", 10)
        self.tool_marker = self.create_publisher(PointStamped, "tool_marker_positon", 10)
        # Create calibration service
        self.create_service(Empty, 'calibrate_tip', self.calibrate_tip_callback)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.bridge = CvBridge()
        #保存board标定结果
        self.tip_calibration_offset = None
        self.calibration_mode = False  # Flag to enable calibration mode


    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        rvecs_list, tvecs_list = [], []
        tool_rvecs_list, tool_tvecs_list = [], []

        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            
            for i, marker_id in enumerate(marker_ids):

                pose = Pose()
                # 设置位置
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]
                # 将旋转向量转换为旋转矩阵
                rot_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                # 将旋转矩阵转换为四元数
                quat = tf3d.quaternions.mat2quat(rot_matrix)
                # 设置四元数
                pose.orientation.x = quat[1]
                pose.orientation.y = quat[2]
                pose.orientation.z = quat[3]
                pose.orientation.w = quat[0]
                # 将当前的姿态添加到PoseArray中
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

                if marker_id[0] in range(1, 5):  # 标定板上的 ArUco 码 ID
                    rvecs_list.append(rvecs[i])
                    tvecs_list.append(tvecs[i])

                elif marker_id[0] in range(10, 16): # 工具上的Aruco码ID
                    tool_rvecs_list.append(rvecs[i])
                    tool_tvecs_list.append(tvecs[i])
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

            # 计算标定板中心点位置
            if self.calibration_mode and rvecs_list and tvecs_list:
            # if  rvecs_list and tvecs_list:
                board_avg_rot_matrix, board_avg_tvec = self.calculate_center(rvecs_list, tvecs_list)
                board_center_position = board_avg_tvec
                if board_center_position is not None:
                    self.publish_tool_tip_position(board_center_position)

                #进行针尖校准
                self.caculate_tip_offset(board_center_position, tool_rvecs_list, tool_tvecs_list)
            else:
            # 实时计算针尖位置
                if self.tip_calibration_offset is not None:
                    tip_position = self.calculate_real_time_tip_position(tool_rvecs_list, tool_tvecs_list)
                    self.publish_tool_tip_position(tip_position)
            

    def calibrate_tip_callback(self, request, response):
        self.get_logger().info("Calibration request received.start calibrating")
        self.calibration_mode = True
        return response
    
    def caculate_tip_offset(self, board_center_position, rvecs_list, tvecs_list):
        """
        校准针尖相对于工具上的 ArUco 码的固定偏移。
        """
        # 确保旋转向量和平移向量的数量相同，并且都大于3
        if len(rvecs_list) <= 3 or len(tvecs_list) <= 3:
            self.get_logger().warn("Calibration requires more than 3 markers.")
            return
        assert len(rvecs_list) == len(tvecs_list), "The number of rotation and translation vectors must be the same"
        # 计算工具中心点位置
        tool_avg_rot_matrix, tool_avg_tvec = self.calculate_center(rvecs_list, tvecs_list)
        self.publish_tool_marker_position(tool_avg_tvec)
        # 计算针尖相对于工具中心的偏移 (工具坐标系下)
        tool_avg_rot_matrix_T = tool_avg_rot_matrix.T  # 工具上 ArUco 码相对于相机的旋转矩阵的转置
        tip_calibration_offset_tool = tool_avg_rot_matrix_T @ (board_center_position - tool_avg_tvec)
        self.tip_calibration_offset = tip_calibration_offset_tool
        self.get_logger().info(f"Calibrated tip offset in tool coordinates: {self.tip_calibration_offset}")
        self.calibration_mode = False

    def calculate_real_time_tip_position(self, rvecs, tvecs):
        """
        根据当前检测到的多个Aruco码的位置和姿态，计算针尖的实时位置。
        """
        if len(rvecs) == 0 or len(tvecs) == 0:
            return None

        num_markers = len(rvecs)
        avg_rot_matrix = np.zeros((3, 3))
        avg_tvec = np.zeros(3)

        # 对所有检测到的Aruco码位置和姿态进行平均
        for rvec, tvec in zip(rvecs, tvecs):
            # 将每个码的旋转向量转换为旋转矩阵，并累加旋转矩阵和平移向量
            rot_matrix = cv2.Rodrigues(rvec)[0]
            avg_rot_matrix += rot_matrix
            # avg_tvec += tvec[0]  
            avg_tvec += tvec.reshape(-1) # 确保tvec是一维的

        # 计算平均旋转矩阵和平均平移向量
        avg_rot_matrix /= num_markers
        avg_tvec /= num_markers

        # 正规化旋转矩阵，确保平均后的矩阵仍然是一个合法的旋转矩阵
        U, _, Vt = np.linalg.svd(avg_rot_matrix)
        avg_rot_matrix = np.dot(U, Vt)

        # 计算针尖的位置 (avg_tvec + avg_rot_matrix * tip_calibration_offset)
        # tip_position = np.dot(avg_rot_matrix, self.tip_calibration_offset) + avg_tvec
        tip_position = avg_tvec + avg_rot_matrix @ np.array(self.tip_calibration_offset) 
        return tip_position

    
    def publish_tool_tip_position(self, tip_position):
        if tip_position is None:
            self.get_logger().warn("未计算出针尖位置；无法发布。")
            return
        point = PointStamped()
        point.header.frame_id = self.camera_frame if self.camera_frame else self.info_msg.header.frame_id
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = float(tip_position[0])
        point.point.y = float(tip_position[1])
        point.point.z = float(tip_position[2])
        self.tip_pub.publish(point)

    def publish_tool_marker_position(self, marker_position):
        if marker_position is None:
            self.get_logger().warn("未计算出针尖位置；无法发布。")
            return
        point = PointStamped()
        point.header.frame_id = self.camera_frame if self.camera_frame else self.info_msg.header.frame_id
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = float(marker_position[0])
        point.point.x = float(marker_position[1])
        point.point.x = float(marker_position[2])
        self.tool_marker.publish(point)

    def calculate_center(self, rvecs, tvecs):
        """
        使用所有标记的旋转和平移向量来计算平均中心点。
        """
        assert len(rvecs) == len(tvecs), "The number of rotation and translation vectors must be the same"
        
        avg_rot_matrix = np.zeros((3, 3), dtype=np.float32)
        avg_tvec = np.zeros(3, dtype=np.float32)
        
        for rvec, tvec in zip(rvecs, tvecs):
            rot_matrix = cv2.Rodrigues(rvec)[0]
            avg_rot_matrix += rot_matrix
            avg_tvec += tvec[0]  # 确保 tvec 是正确的形状
        
        # 对旋转矩阵和平移向量求平均
        num_markers = len(rvecs)
        avg_rot_matrix /= num_markers
        avg_tvec /= num_markers
        
        # 将旋转矩阵正规化以保证其为合法的旋转矩阵
        U, _, Vt = np.linalg.svd(avg_rot_matrix)
        avg_rot_matrix = np.dot(U, Vt)

        # 计算并返回最终的中心位置
        # board_center_world = avg_rot_matrix @ np.array([0.0, 0.0, 0.0]) + avg_tvec
        return avg_rot_matrix, avg_tvec  # 返回旋转矩阵和平移向量





def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
