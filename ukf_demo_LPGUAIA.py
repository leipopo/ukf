# -*- coding: utf-8 -*-
# @Time    : 2023/9/24
# @Author  : LPGUAIA
# @function  : 此项目假定一物体在xy平面运动，设定状态量为x,y,vx,vy,ax,ay，测量量为vx,vy,ax,ay,通过ukf算法估计状态量

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import ukf_func as ukf
import kf_func as kf


# 采样周期
dt = 0.1
time = np.arange(0, 10, dt)
time_1 = np.arange(0, 10 - dt, dt)

# 实际运动状态为x,y,vx,vy,ax,ay
# 运动轨迹为xy平面，沿x轴匀速运动，沿y轴做正弦运动的曲线
# x速度为10m/s，运动时长为10s
x = np.linspace(0, 10 * 10, int(10 / dt))
y = np.sin(x / 10) * 10
vx = np.ones(len(x)) * 10
vy = np.cos(x / 10) * 10
ax = np.zeros(len(x))
ay = -np.sin(x / 10) * 10


# 测量值，加入高斯白噪声
gas_noise_ax = np.random.normal(0, 1, len(ax))
gas_noise_ay = np.random.normal(0, 1, len(ay))
gas_nosie_vx = np.random.normal(0, 1, len(vx))
gas_nosie_vy = np.random.normal(0, 1, len(vy))

# 状态转移矩阵
A = np.array(
    [
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

# 控制矩阵
B = np.array(
    [
        [0.5 * dt * dt, 0],
        [0, 0.5 * dt * dt],
        [dt, 0],
        [0, dt],
        [1, 0],
        [0, 1],
    ]
)

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


# 状态转移方程
def fx(x, u, dt):
    return np.dot(A, x) + np.dot(B, u)


# 测量方程
def hx(x):
    return np.dot(H, x)


fig = plt.figure(figsize=(16, 12))

ax_position_x = fig.add_subplot(3, 2, 1)
ax_position_y = fig.add_subplot(3, 2, 2)
ax_velocity_x = fig.add_subplot(3, 2, 3)
ax_velocity_y = fig.add_subplot(3, 2, 4)
ax_acceleration_x = fig.add_subplot(3, 2, 5)
ax_acceleration_y = fig.add_subplot(3, 2, 6)


fig_witdh = 0.35
fig_height = 0.2
fig_space_x = 0.1
fig_space_y = 0.065
fig_y_offset = 0.03

ax_position_x.set_position(
    [fig_space_x, 1 - fig_space_y - fig_height + fig_y_offset, fig_witdh, fig_height]
)
ax_position_y.set_position(
    [
        fig_space_x + fig_witdh + fig_space_x,
        1 - fig_space_y - fig_height + fig_y_offset,
        fig_witdh,
        fig_height,
    ]
)
ax_velocity_x.set_position(
    [
        fig_space_x,
        1 - 2 * fig_space_y - 2 * fig_height + fig_y_offset,
        fig_witdh,
        fig_height,
    ]
)
ax_velocity_y.set_position(
    [
        fig_space_x + fig_witdh + fig_space_x,
        1 - 2 * fig_space_y - 2 * fig_height + fig_y_offset,
        fig_witdh,
        fig_height,
    ]
)
ax_acceleration_x.set_position(
    [
        fig_space_x,
        1 - 3 * fig_space_y - 3 * fig_height + fig_y_offset,
        fig_witdh,
        fig_height,
    ]
)
ax_acceleration_y.set_position(
    [
        fig_space_x + fig_witdh + fig_space_x,
        1 - 3 * fig_space_y - 3 * fig_height + fig_y_offset,
        fig_witdh,
        fig_height,
    ]
)


ax_v_noise = plt.axes([fig_space_x * 2, 0.17, 1 - fig_space_x * 4, 0.03])
ax_acc_nosis = plt.axes([fig_space_x * 2, 0.12, 1 - fig_space_x * 4, 0.03])
ax_slider_alpha = plt.axes([fig_space_x * 2, 0.07, 1 - fig_space_x * 4, 0.03])
ax_slider_beta = plt.axes([fig_space_x * 2, 0.02, 1 - fig_space_x * 4, 0.03])

slider_v_noise = plt.Slider(ax_v_noise, "v_noise", 0.1, 10, valinit=1, valstep=0.1)
slider_acc_noise = plt.Slider(
    ax_acc_nosis, "acc_noise", 0.1, 10, valinit=1, valstep=0.1
)
slider_alpha = plt.Slider(
    ax_slider_alpha, "alpha", 0.001, 1, valinit=0.1, valstep=0.001
)
slider_beta = plt.Slider(ax_slider_beta, "beta", 0.01, 3, valinit=2, valstep=0.01)


def update(val):
    ukf_aplha = slider_alpha.val
    ukf_beta = slider_beta.val
    v_noise = slider_v_noise.val
    acc_noise = slider_acc_noise.val


slider_alpha.on_changed(update)
slider_beta.on_changed(update)
slider_v_noise.on_changed(update)
slider_acc_noise.on_changed(update)


def animate(i):
    ax_position_x.clear()
    ax_position_y.clear()
    ax_velocity_x.clear()
    ax_velocity_y.clear()
    ax_acceleration_x.clear()
    ax_acceleration_y.clear()

    ax_position_x.set_xlim(0, 10)
    ax_position_y.set_xlim(0, 10)
    ax_velocity_x.set_xlim(0, 10)
    ax_velocity_y.set_xlim(0, 10)
    ax_acceleration_x.set_xlim(0, 10)
    ax_acceleration_y.set_xlim(0, 10)

    ax_position_x.set_ylim(0, 100)
    ax_position_y.set_ylim(-20, 20)
    ax_velocity_x.set_ylim(-10, 30)
    ax_velocity_y.set_ylim(-20, 20)
    ax_acceleration_x.set_ylim(-20, 20)
    ax_acceleration_y.set_ylim(-20, 20)

    ax_position_x.set_title("position_x")
    ax_position_y.set_title("position_y")
    ax_velocity_x.set_title("velocity_x")
    ax_velocity_y.set_title("velocity_y")
    ax_acceleration_x.set_title("acceleration_x")
    ax_acceleration_y.set_title("acceleration_y")

    ax_position_x.set_xlabel("time")
    ax_position_y.set_xlabel("time")
    ax_velocity_x.set_xlabel("time")
    ax_velocity_y.set_xlabel("time")
    ax_acceleration_x.set_xlabel("time")
    ax_acceleration_y.set_xlabel("time")

    ax_position_x.set_ylabel("position_x")
    ax_position_y.set_ylabel("position_y")
    ax_velocity_x.set_ylabel("velocity_x")
    ax_velocity_y.set_ylabel("velocity_y")
    ax_acceleration_x.set_ylabel("acceleration_x")
    ax_acceleration_y.set_ylabel("acceleration_y")

    sli_alpha = slider_alpha.val
    # print(sli_alpha)
    sli_beta = slider_beta.val
    # print(sli_beta)
    sli_v_noise = slider_v_noise.val
    # print(sli_v_noise)
    sli_acc_noise = slider_acc_noise.val

    measure_ax = ax + gas_noise_ax * sli_acc_noise
    measure_ay = ay + gas_noise_ay * sli_acc_noise
    measure_vx = vx + gas_nosie_vx * sli_v_noise
    measure_vy = vy + gas_nosie_vy * sli_v_noise

    # 初始化ukf
    ukf_filter = ukf.UKF(
        dim_x=6,
        dim_z=6,
        dt=dt,
        hx=hx,
        fx=fx,
        alpha=sli_alpha,
        beta=sli_beta,
        kappa=0,
    )
    # 初始化状态量
    ukf_filter.x = np.array([x[0], y[0], vx[0], vy[0], ax[0], ay[0]])
    # 初始化状态协方差矩阵
    ukf_filter.P = np.eye(6)
    # 初始化测量噪声协方差矩阵
    ukf_filter.R = np.eye(6)
    ukf_filter.R[2][2] = sli_v_noise
    ukf_filter.R[3][3] = sli_v_noise
    ukf_filter.R[4][4] = sli_acc_noise
    ukf_filter.R[5][5] = sli_acc_noise
    # 初始化状态噪声协方差矩阵
    ukf_filter.Q = np.eye(6)
    # 初始化ukf结果缓冲区
    ukf_position = np.zeros((2, len(x) - 1))
    ukf_velocity = np.zeros((2, len(x) - 1))
    ukf_acceleration = np.zeros((2, len(x) - 1))

    # 初始化kf
    kf_filter = kf.KF(dim_x=6, dim_z=6, dt=dt, A=A, B=B, H=H)
    # 初始化状态量
    kf_filter.x = np.array([x[0], y[0], vx[0], vy[0], ax[0], ay[0]])
    # 初始化状态协方差矩阵
    kf_filter.P = np.eye(6)
    # 初始化测量噪声协方差矩阵
    kf_filter.R = np.eye(6)
    kf_filter.R[2][2] = sli_v_noise
    kf_filter.R[3][3] = sli_v_noise
    kf_filter.R[4][4] = sli_acc_noise
    kf_filter.R[5][5] = sli_acc_noise
    # 初始化状态噪声协方差矩阵
    kf_filter.Q = np.eye(6)
    # 初始化kf结果缓冲区
    kf_position = np.zeros((2, len(x) - 1))
    kf_velocity = np.zeros((2, len(x) - 1))
    kf_acceleration = np.zeros((2, len(x) - 1))

    # 执行滤波计算
    for i in range(len(x) - 1):
        # ukf
        ukf_filter.predict(u=np.array([ax[i], ay[i]]))
        ukf_filter.update(
            np.array([0, 0, measure_vx[i], measure_vy[i], measure_ax[i], measure_ay[i]])
        )
        # ukf结果缓冲区
        ukf_position[0][i] = ukf_filter.x[0]
        ukf_position[1][i] = ukf_filter.x[1]
        ukf_velocity[0][i] = ukf_filter.x[2]
        ukf_velocity[1][i] = ukf_filter.x[3]
        ukf_acceleration[0][i] = ukf_filter.x[4]
        ukf_acceleration[1][i] = ukf_filter.x[5]

        # kf
        kf_filter.predict(u=np.array([ax[i], ay[i]]))
        kf_filter.update(
            np.array([0, 0, measure_vx[i], measure_vy[i], measure_ax[i], measure_ay[i]])
        )
        # kf结果缓冲区
        kf_position[0][i] = kf_filter.x[0]
        kf_position[1][i] = kf_filter.x[1]
        kf_velocity[0][i] = kf_filter.x[2]
        kf_velocity[1][i] = kf_filter.x[3]
        kf_acceleration[0][i] = kf_filter.x[4]
        kf_acceleration[1][i] = kf_filter.x[5]

    # 绘制图像
    ax_position_x.plot(time, x, label="real", color="red")
    ax_position_x.plot(time_1, ukf_position[0], label="ukf", color="blue")
    ax_position_x.plot(time_1, kf_position[0], label="kf", color="black")
    ax_position_x.legend()

    ax_position_y.plot(time, y, label="real", color="red")
    ax_position_y.plot(time_1, ukf_position[1], label="ukf", color="blue")
    ax_position_y.plot(time_1, kf_position[1], label="kf", color="black")
    ax_position_y.legend()

    ax_velocity_x.plot(time, vx, label="real", color="red")
    ax_velocity_x.plot(time_1, ukf_velocity[0], label="ukf", color="blue")
    ax_velocity_x.plot(time_1, kf_velocity[0], label="kf", color="black")
    ax_velocity_x.plot(time, measure_vx, label="measure", color="green")
    ax_velocity_x.legend()

    ax_velocity_y.plot(time, vy, label="real", color="red")
    ax_velocity_y.plot(time_1, ukf_velocity[1], label="ukf", color="blue")
    ax_velocity_y.plot(time_1, kf_velocity[1], label="kf", color="black")
    ax_velocity_y.plot(time, measure_vy, label="measure", color="green")
    ax_velocity_y.legend()

    ax_acceleration_x.plot(time, ax, label="real", color="red")
    ax_acceleration_x.plot(time_1, ukf_acceleration[0], label="ukf", color="blue")
    ax_acceleration_x.plot(time_1, kf_acceleration[0], label="kf", color="black")
    ax_acceleration_x.plot(time, measure_ax, label="measure", color="green")
    ax_acceleration_x.legend()

    ax_acceleration_y.plot(time, ay, label="real", color="red")
    ax_acceleration_y.plot(time_1, ukf_acceleration[1], label="ukf", color="blue")
    ax_acceleration_y.plot(time_1, kf_acceleration[1], label="kf", color="black")
    ax_acceleration_y.plot(time, measure_ay, label="measure", color="green")
    ax_acceleration_y.legend()


ani = animation.FuncAnimation(fig, animate, interval=50)

plt.show()
