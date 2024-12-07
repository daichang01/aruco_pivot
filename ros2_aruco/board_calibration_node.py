import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data,QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d as tf3d
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_srvs.srv import Empty
# from .pivot_calibration import PivotCalibration
from scipy.optimize import least_squares
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from sensor_msgs_py import point_cloud2  # 用于解析 PointCloud2
import pickle
from .KFClass import KalmanFilterWrapper
from .EKFClass import ExtendedKalmanFilterClass


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
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_250):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))
        

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
       

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.tip_pub = self.create_publisher(PointStamped, "tool_tip_position", 10)
        self.tool_marker = self.create_publisher(PointStamped, "tool_marker_positon", 10)
        self.aruco_image_pub = self.create_publisher(Image, "aruco_image", 10)
        self.trans_image_pub = self.create_publisher(Image, "trans_image", 10)
        self.valpoints_image_pub = self.create_publisher(Image, "valpoints_image", 10)



        # 订阅 /trans_pcd_topic
        self.trans_pcd_sub = self.create_subscription(PointCloud2,'/trans_pcd_topic',self.trans_pcd_callback,10)
        # 订阅 /trans_pcd_point
        self.trans_point_sub = self.create_subscription(PointCloud2,'/trans_pcd_point',self.trans_point_callback,10)
        # Create calibration service
        self.create_service(Empty, 'calibrate_tip', self.calibrate_tip_callback)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        # self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()
        #保存board标定结果
        self.tip_calibration_offset = None
        self.calibration_mode = False  # Flag to enable calibration mode

        self.tip_calibration_offsets = []  # 存储多次采集的偏移量
        self.calibration_samples = 100  # 采集样本数量
        self.current_samples = 0
        self.border_center_position = None #标定板中心位置

        # 初始化 cv_image 为 None
        self.cv_image = None

        # 初始化时尝试加载标定结果
        self.load_calibration_result()
        # Initialize Kalman Filter

        # Initialize Kalman Filter 
        self.kalman_filter = KalmanFilterWrapper()
        # Initialize Extended Kalman Filter
        self.ekf = ExtendedKalmanFilterClass()
        self.dt = 0.1


        # 相机内参
        self.fx = 641.9315185546875
        self.fy = 641.9315185546875
        self.cx = 643.0005493164062
        self.cy = 362.68548583984375

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def save_calibration_result(self, file_path="/home/daichang/Desktop/teeth_ws/src/aruco_pivot/pin_cali_res/calibration_data.pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(self.tip_calibration_offset, f)
        self.get_logger().info(f"Calibration result saved to {file_path}")
    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        # 添加自适应直方图均衡化（CLAHE）来增强图像
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.cv_image = clahe.apply(self.cv_image)

        # 初始化ArucoMarkers和PoseArray消息
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
            self.cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
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
            # 在图像上绘制检测到的 ArUco 标记
            cv2.aruco.drawDetectedMarkers(self.cv_image, corners, marker_ids)
            
            # # 获取当前时间
            # current_time = self.get_clock().now()
            # 先求标定板中心位置
            if len(rvecs_list) == 4 and len(tvecs_list) == 4:
                board_avg_rot_matrix, board_avg_tvec = self.calculate_center(rvecs_list, tvecs_list)
                self.border_center_position = board_avg_tvec
                self.publish_tool_tip_position(self.border_center_position)

        ###################################### 标定与定位 #####################################################
            if self.calibration_mode and self.border_center_position is not None:
                # 标定模式
                #进行针尖校准 （关键步骤）
                
                self.caculate_tip_offset(self.border_center_position, tool_rvecs_list, tool_tvecs_list)
                tip_text = f"Calibration mode"
                cv2.putText(self.cv_image, tip_text, (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # 定位模式
                tip_text = f"Positioning mode"
                cv2.putText(self.cv_image, tip_text, (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # 实时计算针尖位置
                if self.tip_calibration_offset is not None:
                    tip_position = self.calculate_real_time_tip_position(tool_rvecs_list, tool_tvecs_list)
                            # 确保 tip_position 是一个一维数组
                    self.publish_tool_tip_position(tip_position)
                    if tip_position is not None:
                        image_point = self.project_to_image(tip_position)
                        if image_point is not None:
                            # 确保图像点有效，然后在图像上绘制标记
                            cv2.circle(self.cv_image, (int(image_point[0]), int(image_point[1])), 5, (0, 255, 0), -1)
                        else:
                            self.get_logger().warn("Invalid image point, skipping drawing.")
                        # 将针尖位置转换为 Python 浮点数，单位转为毫米 (mm)，并保留两位小数
                        tip_x, tip_y, tip_z = float(tip_position[0]) * 1000, float(tip_position[1]) * 1000, float(tip_position[2]) * 1000
                        tip_text = f"Tip Position: X={tip_x:.2f} mm, Y={tip_y:.2f} mm, Z={tip_z:.2f} mm"
                        cv2.putText(self.cv_image, tip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
        # image_message = self.bridge.cv2_to_imgmsg(self.cv_image, encoding="mono8")
        # self.aruco_image_pub.publish(image_message)
        cv2.imshow("Aruco Image", self.cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            
    def apply_kalman_filter(self, position):
        return self.kalman_filter.predict_and_update(position)


    def apply_ekf(self, tip_position):
        """
        应用扩展卡尔曼滤波器，返回长度为3的smoothed_tip_position
        """
        self.ekf.update(tip_position, self.dt)
        smoothed_tip_position = self.ekf.get_state()

        # 检查 smoothed_tip_position 的维度，并确保其为二维矩阵，保留第一行
        if len(smoothed_tip_position.shape) == 2 and smoothed_tip_position.shape[0] > 0:
            # 只返回第一行
            return smoothed_tip_position[0]
        elif len(smoothed_tip_position.shape) == 1:
            # 如果是 1 维数组，直接返回
            return smoothed_tip_position
        else:
            self.get_logger().warn(f"Unexpected smoothed_tip_position shape: {smoothed_tip_position.shape}")
            return smoothed_tip_position


    def calibrate_tip_callback(self, request, response):
        self.get_logger().info("Calibration request received.start calibrating")
        self.calibration_mode = True
        self.current_samples = 0
        self.tip_calibration_offsets = []
        return response
    
    def trans_pcd_callback(self, msg):
        if self.cv_image is None:
            return

        trans_image = self.cv_image.copy()
        # 将灰度图像转换为 RGB 图像
        trans_image = cv2.cvtColor(trans_image, cv2.COLOR_GRAY2RGB)

        # 解析 PointCloud2 数据
        points = []
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        points = np.array(points)

        # 投影点到图像上
        for point in points:
            u, v = self.project_to_image(point)
            if 0 <= u < trans_image.shape[1] and 0 <= v < trans_image.shape[0]:
                # cv2.circle(trans_image, (int(u), int(v)), 1, (0, 255, 0), -1)
                # 根据深度调整颜色和大小
                color_intensity = min(255, max(0, int(255 * (point[2] / np.max(points[:, 2])))))
                color = (0, 255 - color_intensity, color_intensity)  # 使用渐变颜色
                radius = 1  # 设置点的半径，较小的半径使点更精细
                thickness = -1  # 填充点
                # 绘制圆形点并启用抗锯齿
                cv2.circle(trans_image, (int(u), int(v)), radius, color, thickness, lineType=cv2.LINE_AA)
                            # 设置单个像素点的颜色
                # color_intensity = min(255, max(0, int(255 * (point[2] / np.max(points[:, 2])))))
                # color = (0, 255 - color_intensity, color_intensity)  # 使用渐变颜色
                # trans_image[int(v), int(u)] = color  # 直接修改像素值

        # 显示图像
        # cv2.imshow("Aruco Image with trans", trans_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        image_message = self.bridge.cv2_to_imgmsg(trans_image)
        self.trans_image_pub.publish(image_message)

    def trans_point_callback(self, msg):
        if self.cv_image is None:
            return

        trans_image = self.cv_image.copy()
        # 将灰度图像转换为 RGB 图像
        trans_image = cv2.cvtColor(trans_image, cv2.COLOR_GRAY2RGB)

        # 解析 PointCloud2 数据
        points = []
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        points = np.array(points)

        # 投影点到图像上
        for point in points:
            u, v = self.project_to_image(point)
            if 0 <= u < trans_image.shape[1] and 0 <= v < trans_image.shape[0]:
                cv2.circle(trans_image, (int(u), int(v)), 2, (0, 0, 255), -1)

        # 显示图像
        # cv2.imshow("Aruco Image with val", trans_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        image_message = self.bridge.cv2_to_imgmsg(trans_image)
        self.valpoints_image_pub.publish(image_message)
    
    def load_calibration_result(self, file_path="/home/daichang/Desktop/teeth_ws/src/aruco_pivot/pin_cali_res/calibration_data.pkl"):
        try:
            with open(file_path, "rb") as f:
                self.tip_calibration_offset = pickle.load(f)
                self.get_logger().info(f"tip_calibration_offset: {self.tip_calibration_offset}")
            self.get_logger().info(f"Calibration result loaded from {file_path}")
        except FileNotFoundError:
            self.get_logger().warn(f"No calibration file found at {file_path}, please calibrate the system.")

    
    def caculate_tip_offset(self, board_center_position, rvecs_list, tvecs_list):
        """
        校准针尖相对于工具上的 ArUco 码的固定偏移。
        """
        # 确保旋转向量和平移向量的数量都为6
        if len(rvecs_list) != 6 or len(tvecs_list) != 6:
            tip_text = f"calibration require all 6 markers!"
            cv2.putText(self.cv_image, tip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return
        assert len(rvecs_list) == len(tvecs_list), "The number of rotation and translation vectors must be the same"
        # 计算工具中心点位置
        tool_avg_rot_matrix, tool_avg_tvec = self.calculate_center(rvecs_list, tvecs_list)
        self.publish_tool_marker_position(tool_avg_tvec)
        # 计算针尖相对于工具中心的偏移 (工具坐标系下) 非常重要
        tool_avg_rot_matrix_T = tool_avg_rot_matrix.T  # 工具上 ArUco 码相对于相机的旋转矩阵的转置
        tip_calibration_offset_tool = tool_avg_rot_matrix_T @ (board_center_position - tool_avg_tvec)

        # 存储当前采集的偏移量
        self.tip_calibration_offsets.append(tip_calibration_offset_tool)
        self.current_samples += 1
        self.get_logger().info(f"Collected {self.current_samples} samples/total need {self.calibration_samples}")

        if self.current_samples >= self.calibration_samples:
            # 计算平均偏移量
            avg_tip_calibration_offset = np.mean(self.tip_calibration_offsets, axis=0)
            self.tip_calibration_offset = avg_tip_calibration_offset
            self.get_logger().info(f"Calibrated tip offset in tool coordinates: {self.tip_calibration_offset}！！！！！！！！！")

            #保存标定结果
            self.save_calibration_result()

            # 重置采样计数器和偏移量列表
            self.current_samples = 0
            self.tip_calibration_offsets = []
            self.calibration_mode = False

    def calculate_real_time_tip_position(self, rvecs, tvecs):
        """
        根据当前检测到的多个Aruco码的位置和姿态，计算针尖的实时位置。
        """
        if len(rvecs) == 0 or len(tvecs) == 0:
            return None

        num_markers = len(rvecs)
        if num_markers != 6:
            # self.get_logger().info(f"track require all 6 markers!")
            tip_text = f"track require all 6 markers!"
            cv2.putText(self.cv_image, tip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return None

        avg_rot_matrix, avg_tvec = self.calculate_center(rvecs, tvecs)


        # 正规化旋转矩阵，确保平均后的矩阵仍然是一个合法的旋转矩阵
        U, _, Vt = np.linalg.svd(avg_rot_matrix)
        avg_rot_matrix = np.dot(U, Vt)

        # 计算针尖的位置 (avg_tvec + avg_rot_matrix * tip_calibration_offset)
        # tip_position = np.dot(avg_rot_matrix, self.tip_calibration_offset) + avg_tvec
        tip_position = avg_tvec + avg_rot_matrix @ np.array(self.tip_calibration_offset).reshape(3)
        # Apply Kalman Filter
        # smoothed_tip_position = self.apply_kalman_filter(tip_position)
        # Apply extender kalman filter
        smoothed_tip_position = self.apply_ekf(tip_position)
        if smoothed_tip_position.size == 3:

            return smoothed_tip_position
        else:
            self.get_logger().warn(f"Unexpected smoothed_tip_position size: {smoothed_tip_position.size}")
            return None          

    def remove_outliers(self, data, threshold=3):
        """
        剔除异常值，超过 `threshold` 个标准差的数据将被剔除
        :param data: 输入的标记位置数据 (n个标记的坐标，如平移向量)
        :param threshold: 阈值，默认是3个标准差
        :return: 剔除异常值后的数据
        """
        # 计算均值和标准差
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        
        # 找出绝对距离超过 threshold * 标准差的点（异常值）
        valid_mask = np.all(np.abs(data - mean) <= threshold * std_dev, axis=1)
        
        # 剔除异常值
        cleaned_data = data[valid_mask]
        
        return cleaned_data

    def remove_outliers_rvecs(self, rvecs, threshold=3):
        """
        剔除旋转向量中的异常值
        :param rvecs: 输入的旋转向量列表
        :param threshold: 阈值，默认是3个标准差
        :return: 剔除异常值后的旋转向量
        """
        # 将 rvecs 转换为旋转矩阵
        rotation_matrices = np.array([cv2.Rodrigues(rvec)[0] for rvec in rvecs])

        # 提取旋转矩阵中的分量
        rotation_vectors_flat = rotation_matrices.reshape(len(rvecs), -1)

        # 对旋转矩阵的分量进行剔除异常值处理
        rotation_vectors_cleaned_flat = self.remove_outliers(rotation_vectors_flat, threshold)

        # 将清理后的平坦矩阵重新转换为旋转矩阵
        rotation_matrices_cleaned = rotation_vectors_cleaned_flat.reshape(-1, 3, 3)

        # 将旋转矩阵转换回旋转向量
        rvecs_cleaned = [cv2.Rodrigues(rot_matrix)[0] for rot_matrix in rotation_matrices_cleaned]
        # self.get_logger().info(f"len of cleaned rvecs: {len(rvecs_cleaned)}")

        return rvecs_cleaned
        
    def publish_tool_tip_position(self, tip_position):
        if tip_position is None:
            # self.get_logger().warn("未计算出针尖位置；无法发布。")
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
        point.point.y = float(marker_position[1])
        point.point.z = float(marker_position[2])
        self.tool_marker.publish(point)

    def calculate_center(self, rvecs, tvecs):
        """
        使用所有标记的旋转和平移向量来计算平均中心点。
        """
        assert len(rvecs) == len(tvecs), "The number of rotation and translation vectors must be the same"
        
        avg_rot_matrix = np.zeros((3, 3), dtype=np.float32)
        avg_tvec = np.zeros(3, dtype=np.float32)

        # 对旋转向量进行剔除异常值
        rvecs_cleaned = self.remove_outliers_rvecs(rvecs, threshold=2)
        if len(rvecs_cleaned) < 2:
            return None
        
        for rvec, tvec in zip(rvecs, tvecs):
            # rot_matrix = cv2.Rodrigues(rvec)[0]
            # avg_rot_matrix += rot_matrix
            avg_tvec += tvec[0]  # 确保 tvec 是正确的形状
        
        # 对旋转矩阵和平移向量求平均
        num_markers = len(rvecs)
        # avg_rot_matrix /= num_markers
        avg_tvec /= num_markers
        # 对剔除后的旋转向量进行平均处理
        avg_rot_matrix = np.mean([cv2.Rodrigues(rvec)[0] for rvec in rvecs_cleaned], axis=0)

        
        # 将旋转矩阵正规化以保证其为合法的旋转矩阵
        U, _, Vt = np.linalg.svd(avg_rot_matrix)
        avg_rot_matrix = np.dot(U, Vt)

        # 计算并返回最终的中心位置
        # board_center_world = avg_rot_matrix @ np.array([0.0, 0.0, 0.0]) + avg_tvec
        return avg_rot_matrix, avg_tvec  # 返回旋转矩阵和平移向量

    def project_to_image(self, point):
        """将世界坐标系中的点转换为图像坐标系"""
            # 确保 point 是一个长度为 3 的数组或列表
        if isinstance(point, np.ndarray):
            point = point.flatten()

        if len(point) != 3:
            self.get_logger().warn(f"Invalid point length: {len(point)}")
            return None

        x, y, z = point
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy
        if z <= 0:
            return None
        return (u, v)
    




def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
