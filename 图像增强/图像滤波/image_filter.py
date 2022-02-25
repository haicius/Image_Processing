# Author: 海子
# Create Time: 2021/7/9
# FileName: image_filter
# Descriptions:
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def bi_demo(image):   # 高斯双边模糊，去除无关噪声，保留较好的边缘信息
    dst = cv.bilateralFilter(image, 10, 50, 50)
    #  dst = cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]
    # 高斯双边非线线性滤波,他同时考虑了两像素间的欧氏距离以及灰度值差异，与空间临近度因子，灰度相似度因子相关
    # 因此，邻域中与中心点像素距离最近，灰度更相似的像素赋予较大的权重
    # 参数1：目标图像
    # 参数2:过滤期间的各像素邻域的直径，如果它是非正数，则从sigmaSpace计算。
    # 参数3：色彩空间的sigma参数，该参数较大时，个像素邻域内的相距较远的颜色会被混合到一起
    # 参数4：坐标空间的sigma参数，该参数较大时，只要颜色相近，越远的像素会相互影响
    # 参数5，边界类型
    # 总结：调整两个sigma参数，若都小，则滤波器效果差，太大，导致过度失真（卡通化）；d值，实时应用d=5,严重噪声d=9
    cv.imshow("bi_demo", dst)


def mesh_shift_demo(image):   # 均值漂移算法
    dst = cv.pyrMeanShiftFiltering(image, 0, 15, 15)
    # 第一个参数src，输入图像，8位，三通道的彩色图像，并不要求必须是RGB格式，HSV、YUV等Opencv中的彩色图像格式均可；
    # 第二个参数dst，输出图像，跟输入src有同样的大小和数据格式；
    # 第三个参数sp，定义的漂移物理空间半径大小；
    # 第四个参数sr，定义的漂移色彩空间半径大小；
    # 第五个参数maxLevel，定义金字塔的最大层数；
    # 第六个参数termcrit，定义的漂移迭代终止条件，可以设置为迭代次数满足终止，迭代目标与中心点偏差满足终止，或者两者的结合；
    # 原理：对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，求取该圆形区域内样本的质心，
    # 即密度最大处的点，再以该点为中心继续执行上述迭代过程，直至最终收敛。
    cv.imshow("mesh_demo", dst)


def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gauss_noise_demo(image):   # 添加高斯噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.randint(0, 50, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow("gauss_noise", image)
    dst = cv.GaussianBlur(src, (3, 3), 0)  # 高斯模糊方法
    cv.imshow('Gauss_blur', dst)


def blur_image(image):            # 模糊与滤波操作
    dst1 = cv.blur(image, (3, 3))   # 均值模糊，处理随机噪声效果好，卷积核为3*3的固定值矩阵
    cv.imshow('blur', dst1)
    dst2 = cv.medianBlur(image, 21)  # 中值模糊，对椒盐噪声效果好，参数2为奇数
    cv.imshow('median_blur', dst2)


def custom_blur(image):              # 自定义模糊，抑制声效果，参数2为奇数

    # dst = cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
    # src,原图像  dst,目标图像，与原图像尺寸和通过数相同  ddepth,目标图像的所需深度
    # kernel,卷积核（或相当于相关核），单通道浮点矩阵;如果要将不同的内核应用于不同的通道，请使用拆分将图像拆分为单独的颜色平面，
    # 然后单独处理它们。 anchor,内核的锚点，指示内核中过滤点的相对位置;锚应位于内核中; 默认值（-1，-1）表示锚位于内核中心。
    # detal,在将它们存储在dst中之前，将可选值添加到已过滤的像素中。类似于偏置。
    # borderType,像素外推法，参见BorderTypes
    kernel = np.ones([5, 5], np.float32)/25  # 求卷积核的均值
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)   # 锐化算子，立体感强，边缘细节，图像增强
    # 卷积核算子很重要，不同的算子和图形卷积有不同的效果
    print(kernel)
    dst2 = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow('custom_blur', dst2)


src = cv.imread('crane.jpg', 0)
src = cv.resize(src, dsize=None, fx=0.25, fy=0.25)
cv.namedWindow('input image', cv.WINDOW_NORMAL)
cv.imshow('input image', src)
dst2 = cv.medianBlur(src, 3)
cv.imshow('dst', dst2)
cv.waitKey(0)
cv.destroyAllWindows()  # 释放所有内存

