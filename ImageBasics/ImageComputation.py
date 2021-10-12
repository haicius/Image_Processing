# Author: 海子
# Create Time: 2021/8/3
# FileName: ImageComputation
# Descriptions:  数字图像数学工具

"""
1、区分矩阵运算（点乘）和对应元素运算（无运算符，重要）
2、区分线性运算(满足加法运算律)和非线性运算，
3、算术运算，+，-，*，/
4、逻辑运算，与，或，非
5、空间运算，恒等，平移，旋转，水平剪切，垂直剪切，伽马运算
6、图像变换，例如图像平面空间--->傅里叶变换--->频域--->逆傅里叶变换--->图像平面空间
7、。。。
"""
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
# from noise import a
from pkg.noise import a


print(a)
"""
算数运算
"""


def get_image_ifo(image):  # 访问图片信息
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_date = np.array(image)
    print(pixel_date)


def math_demo(image1, image2):  # +,-,*,/
    add_image1 = image1 + image2  # numpy 加法，运算结果超出255进行取模运算
    cv.imshow("numpy_add_image", add_image1)

    add_image2 = cv.add(image1, image2)
    cv.imshow("opencv_add_image", add_image2)  # opencv 加法，运算结果超出255进行截断

    cv.addWeighted(image1, 0.5, image2, 0.5, 0)  # 图片融合
    cv.imshow('add_demo', cv.add(image1, image2))

    cv.imshow('subtract_demo', cv.subtract(image1, image2))  # 减法运算

    cv.imshow('divide_demo', cv.divide(image1, image2))  # 除法运算

    cv.imshow('multiply_demo', cv.multiply(image1, image2))  # 乘法运算

    mean_a = cv.mean(image1)  # 均值
    print(mean_a)

    mean_b = cv.mean(image2)
    print(mean_b)

    std_a = cv.meanStdDev(image1)    # 返回均值和方差
    print(std_a)

    mean_c, std_b = cv.meanStdDev(image2)
    print(mean_c)
    print(std_b)

    h, w = image1.shape[:2]

    img = np.zeros([h, w], np.uint8)  # 创建一个512*512的0值图像
    std_a = cv.meanStdDev(img)
    cv.imshow('img', img)
    print(std_a)


def logic_demo(m1, m2):  # 像素逻辑运算,遮罩层应用
    cv.imshow('logic_and_demo', cv.bitwise_and(m1, m2))
    cv.imshow('logic_not_demo', cv.bitwise_not(m1))
    cv.imshow('logic_or_demo', cv.bitwise_or(m1, m2))


def contrast_bright_demo(image, c, b):  # 图像对比度增强，亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.imshow('con_bri_demo', dst)


def access_pixeles(image):  # 访问图片像素，BGR图像对数，伽马运算
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    k = 1
    gamma = 0.8  # gamma值越大，输出图像越暗
    print('height = %s, width = %s, channels = %s' % (height, width, channels))
    for h in range(height):  # 图像反转，对数变换，幂律（伽马y）变换
        for w in range(width):
            for c in range(channels):
                pv = image[h, w, c]
                # image[h, w, c] = 255 - pv  # 图像反转
                # image[h, w, c] = c * (math.log(pv+1.0))   # 图像对数变换
                image[h, w, c] = k * (math.pow(pv, gamma))  # 图像伽马变换
    image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
    imgge = cv.convertScaleAbs(image)
    cv.imshow('pixels_demo', image)


def nonlinear_transformation(image):    # 非线性变换out=in*in/255
    result = np.zeros(image.shape, np.uint8)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            image1 = int(image[i, j])*int(image[i, j]) / 255
            result[i, j] = np.uint8(image1)
    cv.imshow("nonliner_trans", result)


def log(c, image):    # 对数变换
    output = c * np.log(1.0 + image)
    output = np.uint8(output+0.5)
    cv.imshow("output", output)


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

#绘制曲线
def gamma_plot(c, v):
    x = np.arange(0, 256, 0.01)
    y = c*x**v
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
    plt.title(u'伽马变换函数')
    plt.xlim([0, 255]), plt.ylim([0, 255])
    plt.show()


# 伽玛变换
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)   # 像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    cv.imshow("output", output_img)


def add_mean_demo(image):
    result = np.zeros(image.shape, np.uint8)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            image1 = int(image[i, j]) * 1000 / 1000
            result[i, j] = np.uint8(image1)
    cv.imshow("add_demo", result)


def create_image():  # 创建一个8位的RGB图像(blue,green,red)以及单通道的灰度图
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    img[:, :, 2] = np.ones([400, 400]) * 255
    cv.imshow('new window', img)
    img = np.zeros([400, 400, 1], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 127
    m2 = img.reshape([200, 800])  # 保持size不变，对整个数字图像进行重新排布
    m3 = cv.resize(img, (300, 300))  # 对整个数字图像进行缩放操作，降采样和升采样
    cv.imshow('new window', img)


src = cv.imread('Image/lena.png', 0)
img1 = cv.imread('Image/gauss.noise.jpg', 0)
img2 = cv.imread('Image/sp.noise.jpg', 1)
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('input image', src)

t1 = cv.getTickCount()  # 时钟函数，单位s
gamma(src, 0.00000005, 4.0)
t2 = cv.getTickCount()

time = (t2 - t1) / cv.getTickFrequency()  # 计算图片转化时间
print('time-->%s ms' % (time * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
