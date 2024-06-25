import numpy as np
import transforms3d as tf3d
from geometry_msgs.msg import Pose, PoseArray
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