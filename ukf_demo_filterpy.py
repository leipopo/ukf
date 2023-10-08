# -*- coding: utf-8 -*-
# @Time    : 2023/9/24
# @Author  : LPGUAIA
# @function  : 此项目假定一物体在xy平面运动，设定状态量为x,y,vx,vy,ax,ay，测量量为vx,vy,ax,ay,通过ukf算法估计状态量

import numpy as np
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSSP

# import ukf_func as ukf

# 采样周期
dt = 0.1

# 实际运动状态为x,y,vx,vy,ax,ay
# 运动轨迹为xy平面，沿x轴匀速运动，沿y轴做正弦运动的曲线
# x速度为10m/s，运动时长为10s
x = np.linspace(0, 10 * 10, int(10 / 0.1))
y = np.sin(x / 10) * 10
vx = np.ones(len(x)) * 10
vy = np.cos(x / 10) * 10
ax = np.zeros(len(x))
ay = -np.sin(x / 10) * 10


# 测量值，加入高斯白噪声
gas_noise_ax = np.random.normal(0, 1, len(ax))
gas_noise_ay = np.random.normal(0, 1, len(ay))
measure_ax = ax + gas_noise_ax
measure_ay = ay + gas_noise_ay
gas_nosie_vx = np.random.normal(0, 1, len(vx))
gas_nosie_vy = np.random.normal(0, 1, len(vy))
measure_vx = vx + gas_nosie_vx
measure_vy = vy + gas_nosie_vy


# # 状态转移方程
# def fx(x, u, dt):
#     # 状态转移矩阵
#     A = np.array(
#         [
#             [1, 0, dt, 0, 0, 0],
#             [0, 1, 0, dt, 0, 0],
#             [0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0],
#         ]
#     )

#     # 控制矩阵
#     B = np.array(
#         [
#             [0.5 * dt * dt, 0],
#             [0, 0.5 * dt * dt],
#             [dt, 0],
#             [0, dt],
#             [1, 0],
#             [0, 1],
#         ]
#     )
#     return np.dot(A, x) + np.dot(B, u)


# 状态转移方程
def fx(x, dt):
    # 状态转移矩阵
    A = np.array(
        [
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    return np.dot(A, x)


# 测量方程
def hx(x):
    H = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    return np.dot(H, x)


ukf = UKF(
    dim_x=6,
    dim_z=6,
    dt=dt,
    hx=hx,
    fx=fx,
    points=MSSP(6, alpha=0.01, beta=2, kappa=0),
)
ukf.x = np.array(
    [x[0], y[0], measure_vx[0], measure_vy[0], measure_ax[0], measure_ay[0]]
)
ukf.P = np.eye(6)
ukf.R = np.eye(6)
ukf.Q = np.eye(6)

ukf.update(np.array([0, 0, measure_vx[0], measure_vy[0], measure_ax[0], measure_ay[0]]))

ukf_position = np.zeros((2, len(x) - 1))
ukf_velocity = np.zeros((2, len(x) - 1))
ukf_acceleration = np.zeros((2, len(x) - 1))

for i in range(0, len(x) - 1):
    ukf.predict()
    ukf.update(
        np.array([0, 0, measure_vx[i], measure_vy[i], measure_ax[i], measure_ay[i]])
    )
    ukf_position[0][i] = ukf.x[0]
    ukf_position[1][i] = ukf.x[1]
    ukf_velocity[0][i] = ukf.x[2]
    ukf_velocity[1][i] = ukf.x[3]
    ukf_acceleration[0][i] = ukf.x[4]
    ukf_acceleration[1][i] = ukf.x[5]


fig = plt.figure(figsize=(16, 10))

ax_position_x = fig.add_subplot(3, 2, 1)
ax_position_y = fig.add_subplot(3, 2, 2)
ax_velocity_x = fig.add_subplot(3, 2, 3)
ax_velocity_y = fig.add_subplot(3, 2, 4)
ax_acceleration_x = fig.add_subplot(3, 2, 5)
ax_acceleration_y = fig.add_subplot(3, 2, 6)

ax_position_x.plot(x, label="real", color="red")
ax_position_x.plot(ukf_position[0], label="ukf", color="blue")
ax_position_x.legend()

ax_position_y.plot(y, label="real", color="red")
ax_position_y.plot(ukf_position[1], label="ukf", color="blue")
ax_position_y.legend()

ax_velocity_x.plot(vx, label="real", color="red")
ax_velocity_x.plot(ukf_velocity[0], label="ukf", color="blue")
ax_velocity_x.plot(measure_vx, label="measure", color="green")
ax_velocity_x.legend()

ax_velocity_y.plot(vy, label="real", color="red")
ax_velocity_y.plot(ukf_velocity[1], label="ukf", color="blue")
ax_velocity_y.plot(measure_vy, label="measure", color="green")
ax_velocity_y.legend()

ax_acceleration_x.plot(ax, label="real", color="red")
ax_acceleration_x.plot(ukf_acceleration[0], label="ukf", color="blue")
ax_acceleration_x.plot(measure_ax, label="measure", color="green")
ax_acceleration_x.legend()

ax_acceleration_y.plot(ay, label="real", color="red")
ax_acceleration_y.plot(ukf_acceleration[1], label="ukf", color="blue")
ax_acceleration_y.plot(measure_ay, label="measure", color="green")
ax_acceleration_y.legend()

plt.show()
plt.waitforbuttonpress(0)
