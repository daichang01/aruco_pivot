
from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanFilterWrapper:
    def __init__(self):
        # Initialize Kalman Filter
        # 设置状态向量维度 dim_x 为6，观测向量维度 dim_z 为3。
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        # F 为状态转移矩阵，这里设置为6x6单位矩阵，意味着状态变量各自独立变化。
        self.kf.F = np.eye(6)  # State transition matrix
        # H 为观测矩阵，它将状态向量映射到观测空间，这里前3个状态变量直接被观测到，后3个状态变量不被直接观测。
        self.kf.H = np.hstack([np.eye(3), np.zeros((3, 3))])  # Measurement function
        # P 为初始估计误差协方差矩阵，较大的值（如1000）表示对初态不确定性较高。
        self.kf.P *= 1000.  # Initial uncertainty
        # R 为观测噪声协方差矩阵，较小的值（如0.01）表示观测相对准确。
        self.kf.R = np.eye(3) * 0.01  # Measurement noise
        # Q 为过程噪声协方差矩阵，较小的值（如0.01）表示系统动态变化平缓。
        self.kf.Q = np.eye(6) * 0.001  # Process noise
        # x 的前3个元素设为0，表示初始状态（如位置）为原点；后3个元素设为0，表示初始速度为0。
        self.kf.x[:3] = 0  # Initial state (assuming the needle starts at the origin)
        self.kf.x[3:] = 0  # Initial velocity

    def predict_and_update(self, position):
        """Predict and update Kalman filter with the current position measurement."""
        self.kf.predict()
        self.kf.update(position)
        return self.kf.x[:3]
