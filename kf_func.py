# -*- coding: utf-8 -*-
# @Time    : 2023/9/24
# @Author  : LPGUAIA
# @function  : 卡尔曼滤波器，对多维状态进行估计

import numpy as np


class KF(object):
    def __init__(self, dim_x, dim_z, dt, A, B, H):
        """
        dim_x: 状态量维度
        dim_z: 测量量维度
        dt: 采样时间
        A,B,H: 状态转移矩阵，控制矩阵，测量矩阵
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.A = A
        self.B = B
        self.H = H

        # 状态量
        self.x = np.zeros(dim_x)
        # 测量量
        self.z = np.zeros(dim_z)
        # 状态协方差矩阵
        self.P = np.eye(dim_x)
        # 测量噪声协方差矩阵
        self.R = np.eye(dim_z)
        # 状态噪声协方差矩阵
        self.Q = np.eye(dim_x)

    def predict(self, u):
        """
        预测
        """
        # 预测状态
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        # 预测协方差
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        更新
        """
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # 更新状态
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        # 更新协方差
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
