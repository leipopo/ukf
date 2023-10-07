# -*- coding: utf-8 -*-
# @Time    : 2023/9/24
# @Author  : LPGUAIA
# @function  : 无迹卡尔曼滤波器，对多维状态进行估计

import numpy as np


class UKF(object):
    def __init__(self, dim_x, dim_z, dt, hx, fx, alpha=1e-1, beta=2, kappa=0):
        """
        dim_x: 状态量维度
        dim_z: 测量量维度
        dt: 采样时间
        hx: 测量函数
        fx: 状态转移函数
        alpha, beta, kappa: ukf参数
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.hx = hx
        self.fx = fx
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

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

        # 测量权重
        self.Wm = np.zeros(2 * dim_x + 1)
        # 协方差权重
        self.Wc = np.zeros(2 * dim_x + 1)

    def sigma_points(self):
        """
        生成sigma点
        """
        # 状态量维度
        n = self.dim_x
        # sigma点个数
        nsig = 2 * n + 1

        # 权重计算
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wc[0] = lambda_ / (n + lambda_) + 1 - self.alpha**2 + self.beta
        for i in range(1, nsig):
            self.Wm[i] = 1 / (2 * (n + lambda_))
            self.Wc[i] = 1 / (2 * (n + lambda_))

        # sigma点矩阵
        sigmas = np.zeros((nsig, n))

        # sigma点协方差矩阵开方
        Psqrt = np.linalg.cholesky((n + lambda_) * self.P)

        # sigma点生成
        sigmas[0] = self.x
        for i in range(n):
            sigmas[i + 1] = self.x + Psqrt[i]
            sigmas[n + i + 1] = self.x - Psqrt[i]

        return sigmas, self.Wm, self.Wc

    def predict(self, u=0):
        """
        预测
        """
        # 生成sigma点
        sigmas_x, Wm, Wc = self.sigma_points()
        # sigma点预测
        for i in range(len(sigmas_x)):
            sigmas_x[i] = self.fx(sigmas_x[i], u, self.dt)
        # 状态量预测
        self.x = np.zeros(self.dim_x)
        for i in range(len(sigmas_x)):
            self.x += sigmas_x[i] * Wm[i]

        self.P = self.Q
        # 协方差矩阵预测
        for i in range(len(sigmas_x)):
            self.P += (self.x - sigmas_x[i]).dot((self.x - sigmas_x[i]).T) * Wc[i]
        return self.x, self.P

    def update(self, z):
        """
        更新
        """
        # 生成sigma点
        sigmas, Wm, Wc = self.sigma_points()
        # sigma点预测
        sigmas_z = np.zeros((len(sigmas), self.dim_z))
        for i in range(len(sigmas)):
            sigmas_z[i] = self.hx(sigmas[i])
        # 测量量预测
        self.z = np.zeros(self.dim_z)
        for i in range(len(sigmas)):
            self.z += sigmas_z[i] * Wm[i]

        self.P = self.R
        # 测量量协方差矩阵预测
        for i in range(len(sigmas)):
            self.P += (sigmas_z[i] - self.z).dot((sigmas_z[i] - self.z).T) * Wc[i]
        # 测量量与状态量协方差矩阵预测
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigmas)):
            Pxz += (sigmas[i] - self.x).dot((sigmas_z[i] - self.z).T) * Wc[i]
        # 卡尔曼增益
        K = np.dot(Pxz, np.linalg.inv(self.P))
        # 状态量更新
        self.x += np.dot(K, z - self.z)
        # 协方差矩阵更新
        self.P -= np.dot(K, Pxz.T)
        return self.x, self.P
