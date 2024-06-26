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

class PivotCalibration:
    def __init__(self, min_poses = 20):
        self.calibration_poses = []
        self.min_poses = min_poses  # 最少需要的姿态数量

    def add_pose(self, pose: Pose):
        self.calibration_poses.append(pose)
    def add_poses(self, pose_array: PoseArray):
        self.calibration_poses.extend(pose_array.poses)
        # print(f"Added new poses. Total poses: {len(self.calibration_poses)}")

    def calibrate(self):
        """
        Perform pivot calibration to find the tool tip position.
        """
        if len(self.calibration_poses) < self.min_poses:
            print("Not enough poses for calibration!")
            return None

        t_matrices = []
        for pose in self.calibration_poses:
            t_matrix = tf3d.affines.compose(
                T=np.array([pose.position.x, pose.position.y, pose.position.z]),
                R=tf3d.quaternions.quat2mat([
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z
                ]),  # 将四元数转换为旋转矩阵
                Z=np.ones(3) # 缩放因子设置为1
            )
            t_matrices.append(t_matrix) # 将变换矩阵添加到列表中

        # 计算枢轴点（工具尖端位置）
        A = np.zeros((3, 3)) # 初始化A矩阵
        B = np.zeros(3) # 初始化B向量
        for t in t_matrices:
            R = t[:3, :3]
            T = t[:3, 3]
            A += R.T @ R # 计算A矩阵的累加值
            B += R.T @ T # 计算B向量的累加值
        try:
            tip_position = np.linalg.inv(A) @ B  # 求解线性方程组，得到尖端位置
            print(f"Calculated tool tip position: {tip_position}")  # 打印计算出的工具尖端位置
        except np.linalg.LinAlgError:
            print("Failed to invert matrix A. Calibration failed.")
            return None

        return tip_position
    
    def calibrate_linalg(self):
        if len(self.calibration_poses) < self.min_poses:
            print("Not enough poses for calibration!")
            return None

        t_matrices = [self.pose_to_transform(pose) for pose in self.calibration_poses]

        A = sum(R.T @ R for R, T in t_matrices)
        B = sum(R.T @ T for R, T in t_matrices)

        try:
            tip_position = np.linalg.solve(A, B)
            print(f"Calculated tool tip position: {tip_position}")
            return tip_position
        except np.linalg.LinAlgError:
            print("Failed to invert matrix A. Calibration failed.")
            return None


    @staticmethod
    def pose_to_transform(pose):
        R = tf3d.quaternions.quat2mat([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        T = np.array([pose.position.x, pose.position.y, pose.position.z])
        return R, T

    def calibrate_least_squares(self):
        """
        Perform pivot calibration to find the tool tip position.
        """
        if len(self.calibration_poses) < self.min_poses:
            print("Not enough poses for calibration!")
            return None

        t_matrices = []
        for pose in self.calibration_poses:
            t_matrix = tf3d.affines.compose(
                T=np.array([pose.position.x, pose.position.y, pose.position.z]),
                R=tf3d.quaternions.quat2mat([
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z
                ]),  # 将四元数转换为旋转矩阵
                Z=np.ones(3)  # 缩放因子设置为1
            )
            t_matrices.append(t_matrix)  # 将变换矩阵添加到列表中

        def residuals(tip_position, t_matrices):
            res = []
            for t in t_matrices:
                R = t[:3, :3]
                T = t[:3, 3]
                estimated_tip = R @ tip_position + T
                res.append(estimated_tip - T)
            return np.array(res).flatten()

        initial_guess = np.array([0.01, 0.01, 0.01])  # 初始猜测为接近原点的一个小值
        result = least_squares(residuals, initial_guess, args=(t_matrices,))

        if result.success:
            tip_position = result.x
            print(f"Calculated tool tip position: {tip_position}")
        else:
            print("Calibration failed.")
            return None

        return tip_position
    


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
        # Create an instance of PivotCalibration
        self.pivot_calibration = PivotCalibration(min_poses=20)

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.tip_pub = self.create_publisher(PointStamped, "tool_tip_position", 10)
        # Create calibration service
        self.create_service(Empty, 'calibrate_pivot', self.calibrate_pivot_callback)
        
        self.pose_array = PoseArray()

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.bridge = CvBridge()
        #保存枢轴标定结果
        self.tip_calibration_offset = None

        # 定时器在初始化时不启动
        self.timer = None


    def start_timer(self):
        if self.timer is None:
            self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz
    def stop_timer(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
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
        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            
            reference_pose = None
            for i, marker_id in enumerate(marker_ids):
                # if marker_id[0] < 10 or marker_id[0] > 15:
                # # 过滤掉不需要的码
                #     continue
                if marker_id[0] >= 10 and marker_id[0] <= 15:
                    pose = Pose()
                    pose.position.x = tvecs[i][0][0]
                    pose.position.y = tvecs[i][0][1]
                    pose.position.z = tvecs[i][0][2]

                    rot_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                    quat = tf3d.quaternions.mat2quat(rot_matrix)

                    pose.orientation.x = quat[1]
                    pose.orientation.y = quat[2]
                    pose.orientation.z = quat[3]
                    pose.orientation.w = quat[0]

                    pose_array.poses.append(pose)
                    markers.poses.append(pose)
                    markers.marker_ids.append(marker_id[0])

            self.pose_array = pose_array
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)
            # 实时计算针尖位置
            if self.tip_calibration_offset is not None:
                tip_position = self.calculate_real_time_tip_position(rvecs, tvecs)
                self.publish_tool_tip_position(tip_position)
            


    def calibration_callback(self, pose_array:PoseArray):       
        # 计算加权平均姿态
        average_pose = self.calculate_average_pose(pose_array)
        self.pivot_calibration.add_pose(average_pose)
        # self.pivot_calibration.add_poses(pose_array)
        self.get_logger().info(f"Collected {len(self.pivot_calibration.calibration_poses)} poses for calibration.")
        if len(self.pivot_calibration.calibration_poses) >= self.pivot_calibration.min_poses:
                self.get_logger().info("Collected enough poses for calibration.")
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None
                self.stop_timer()
                # tip_position = self.pivot_calibration.calibrate_least_squares()
                # tip_position = self.pivot_calibration.calibrate()
                tip_position = self.pivot_calibration.calibrate_linalg()
                if tip_position is not None:
                    self.tip_calibration_offset = tip_position
                    self.get_logger().info(f"Tool tip position: {tip_position}")

    def calculate_average_pose(self, pose_array: PoseArray):
        num_poses = len(pose_array.poses)
        if num_poses == 0:
            return None
        
        # 计算位置的平均值
        avg_position = np.mean([[pose.position.x, pose.position.y, pose.position.z] for pose in pose_array.poses], axis=0)

        # 计算四元数的平均值
        quaternions = np.array([[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] for pose in pose_array.poses])
        avg_quaternion = self.average_quaternions(quaternions)

        # 创建平均姿态
        avg_pose = Pose()
        avg_pose.position.x = avg_position[0]
        avg_pose.position.y = avg_position[1]
        avg_pose.position.z = avg_position[2]
        avg_pose.orientation.w = avg_quaternion[0]
        avg_pose.orientation.x = avg_quaternion[1]
        avg_pose.orientation.y = avg_quaternion[2]
        avg_pose.orientation.z = avg_quaternion[3]

        return avg_pose

    def average_quaternions(self, quaternions):
        """
        Compute the average quaternion.
        Reference: Markley, F. Landis, et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
        """
        A = np.zeros((4, 4))
        for q in quaternions:
            A += np.outer(q, q)
        A /= len(quaternions)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        avg_quaternion = eigenvectors[:, np.argmax(eigenvalues)]
        return avg_quaternion

    def timer_callback(self):
        self.calibration_callback(self.pose_array)
    def calibrate_pivot_callback(self, request, response):
        self.get_logger().info("Calibration request received. Starting data collection.")
        self.start_timer()
        return response
    
    # def calculate_real_time_tip_position(self, rvecs, tvecs):
    #     # 根据当前的Aruco码位置和姿态，计算针尖的实时位置
    #     if len(rvecs) == 0 or len(tvecs) == 0:
    #         return None
    #     # 假设使用第一个检测到的 ArUco 码的位置和姿态来计算针尖位置
    #     rvec = rvecs[0]
    #     tvec = tvecs[0]

    #     # 将旋转向量转换为旋转矩阵
    #     rot_matrix = cv2.Rodrigues(rvec)[0]

    #     # 计算针尖的位置（tvec + rot_matrix * tip_calibration_offset）
    #     tip_position = np.dot(rot_matrix, self.tip_calibration_offset) + tvec[0]

    #     return tip_position
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
            rot_matrix, _ = cv2.Rodrigues(rvec)
            avg_rot_matrix += rot_matrix
            avg_tvec += tvec.reshape(3)

        # 计算平均旋转矩阵和平均平移向量
        avg_rot_matrix /= num_markers
        avg_tvec /= num_markers

        # 计算针尖的位置 (avg_tvec + avg_rot_matrix * tip_calibration_offset)
        tip_position = np.dot(avg_rot_matrix, self.tip_calibration_offset) + avg_tvec

        return tip_position
    
    def publish_tool_tip_position(self, tip_position):
        point = PointStamped()
        point.header.frame_id = self.camera_frame if self.camera_frame else self.info_msg.header.frame_id
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = tip_position[0]
        point.point.y = tip_position[1]
        point.point.z = tip_position[2]
        self.tip_pub.publish(point)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
