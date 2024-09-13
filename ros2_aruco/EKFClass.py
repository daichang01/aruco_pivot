import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class ExtendedKalmanFilterClass:
    def __init__(self):
        # 初始化 EKF，状态维度为6，观测维度为3
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        
        # 初始状态 (位置和速度)
        self.ekf.x[:3] = 0  # 初始位置
        self.ekf.x[3:] = 0  # 初始速度

        # 初始协方差矩阵 P
        self.ekf.P *= 500  # 初始状态不确定性大
        self.ekf.R = np.eye(3) * 0.005  # 观测噪声
        self.ekf.Q = np.eye(6) * 0.0001  # 过程噪声，调节此参数以获得更平滑的估计

    def state_transition_function(self, x, dt):
        """
        状态转移方程，计算下一时刻的状态
        x: 当前状态向量
        dt: 时间间隔
        """
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return np.dot(F, x)

    def measurement_function(self, x):
        """
        观测方程，返回从状态到观测的映射（直接观测针尖位置）
        x: 状态向量
        返回：观测向量 (x, y, z)
        """
        return x[:3]

    def jacobian_of_transition(self, x, dt):
        """
        状态转移方程的雅可比矩阵
        x: 状态向量
        dt: 时间步长
        """
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def jacobian_of_measurement(self, x):
        """
        观测方程的雅可比矩阵，观测为直接获取位置，雅可比矩阵为 (3x6)
        """
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        return H

    # def update(self, z, dt):
    #     """
    #     使用 EKF 更新滤波器状态
    #     z: 观测位置 (x, y, z)
    #     dt: 时间间隔
    #     """
    #     self.ekf.predict_update(
    #         z,
    #         HJacobian=self.jacobian_of_measurement,
    #         Hx=self.measurement_function,
    #         F=self.jacobian_of_transition,
    #         fx=self.state_transition_function,
    #         args=(dt,),
    #         hx_args=(),
    #     )

    def update(self, z, dt):
        """
        使用 EKF 更新滤波器状态
        z: 观测位置 (x, y, z)
        dt: 时间间隔
        """
        # 设置状态转移矩阵 F
        self.ekf.F = self.jacobian_of_transition(self.ekf.x, dt)
        
        # 设置观测矩阵 H
        self.ekf.H = self.jacobian_of_measurement(self.ekf.x)

        # 调用 predict() 进行状态预测
        self.ekf.predict()

        # 调用 update() 进行观测更新
        self.ekf.update(z, HJacobian=self.jacobian_of_measurement, Hx=self.measurement_function)


    def get_state(self):
        """
        获取滤波后的状态向量
        返回：位置 (x, y, z)
        """
        return self.ekf.x[:3]
