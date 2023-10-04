# -*- coding: utf-8 -*-
# @Time    : 2023/9/24
# @Author  : LPGUAIA
# @function  : 此项目假定一物体在xy平面运动，设定状态量为x,y,vx,vy,ax,ay，测量量为vx,vy,ax,ay,通过ukf算法估计状态量

import numpy as np
import matplotlib.pyplot as plt

# from filterpy.kalman import UnscentedKalmanFilter as UKF
import ukf_func as ukf

# 采样周期
T = 0.001

# 实际运动状态为x,y,vx,vy,ax,ay
# 运动轨迹为xy平面，沿x轴匀速运动，沿y轴做正弦运动的曲线
# x速度为10m/s，运动时长为10s
x = np.linspace(0, 10 * 10, 10000)
y = np.sin(x / 10) * 10
vx = np.ones(len(x)) * 10
vy = np.cos(x / 10) * 10
ax = np.zeros(len(x))
ay = -np.sin(x / 10) * 10


# 测量值，加入高斯白噪声
gas_noise_ax = np.random.normal(0, 1, len(ax))
gas_noise_ay = np.random.normal(0, 0.1, len(ay))
measure_ax = ax + gas_noise_ax
measure_ay = ay + gas_noise_ay
gas_nosie_vx = np.random.normal(0, 1, len(vx))
gas_nosie_vy = np.random.normal(0, 1, len(vy))
measure_vx = vx + gas_nosie_vx
measure_vy = vy + gas_nosie_vy

# # 画图
# plt.figure()
# plt.plot(x, y, label="real")
# plt.plot(x, measure_vx, label="measure_vx")
# plt.plot(x, measure_vy, label="measure_vy")
# plt.plot(x, measure_ax, label="measure_ax")
# plt.plot(x, measure_ay, label="measure_ay")
# plt.show()
# plt.waitforbuttonpress(0)

# 测量矩阵
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

# 状态转移矩阵
A = np.array(
    [
        [1, 0, T, 0, 0, 0],
        [0, 1, 0, T, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

# 控制矩阵
B = np.array(
    [
        [0.5 * T * T, 0],
        [0, 0.5 * T * T],
        [T, 0],
        [0, T],
        [1, 0],
        [0, 1],
    ]
)


# 状态转移方程
def fx(x, u, dt):
    x = np.dot(A, x) + np.dot(B, u)
    return x


# 测量方程
def hx(x):
    z = np.dot(H, x)
    return z


ukf = ukf.UKF(dim_x=6, dim_z=6, dt=T, hx=hx, fx=fx)
ukf.x = np.array(
    [x[0], y[0], measure_vx[0], measure_vy[0], measure_ax[0], measure_ay[0]]
)
ukf.P = np.eye(6) * 0.1
ukf.R = np.eye(6) * 0.1
ukf.Q = np.eye(6) * 0.1

for i in range(0, len(x) - 1):
    ukf.predict(np.array([ax[i], ay[i]]))
    ukf.update(
        np.array([0, 0, measure_vx[i], measure_vy[i], measure_ax[i], measure_ay[i]])
    )

    print(i)
