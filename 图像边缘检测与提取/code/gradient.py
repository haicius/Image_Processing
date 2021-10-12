# Author: 海子
# Create Time: 2021/7/14
# FileName: gradient
# Descriptions: 图像梯度运算，求边缘后可继续进行二值化处理
# 一阶导数：Soble算子以及scharr算子具有一阶导数最大值，包括水平梯度Gx和垂直梯度Gy  总图像梯度G = |Gx|+|Gy|
#         Gy = [-1 -2 -1, 0 0 0, 1 2 1], Gx =[-1 0 1, -2 0 2, -1 0 1]

# 二阶导数：拉普拉斯算子，在二阶导数的时候，最大变化处的值为0即边缘是零值，计算图像二阶导数求边缘
#          L = [0 1 0, 1 -4, 1, 0 1 0], 或者= [1 1 1, 1 -8 1, 1 1 1]
import cv2 as cv
import numpy as np


# 拉普拉斯算子
def laplacian_demo(image):     #  默认第一种算子
    """
    void cv::convertScaleAbs(
    cv::InputArray src, // 输入数组
    cv::OutputArray dst, // 输出数组
    double alpha = 1.0, // 乘数因子
    double beta = 0.0 // 偏移量
         );
    dst = |src*alpha + beta|
    """
    laplacian_xy = cv.Laplacian(image, cv.CV_32F)
    laplacian_xy = cv.convertScaleAbs(laplacian_xy, )
    cv.imshow("laplacian", laplacian_xy)


# 自定义拉普拉斯算子
def custom_laplacian(image):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image, cv.CV_32F, kernel)
    laplacian_xy = cv.convertScaleAbs(dst)
    cv.imshow("laplacian", laplacian_xy)


# sobel，scharr算子梯度运算,
def sobel_demo(image):
    gradient_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    gradient_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradient_x = cv.convertScaleAbs(gradient_x)    # 参数1输入数组，参数alpha乘数因子(默认为1，beta偏移量（0），求绝对值
    gradient_y = cv.convertScaleAbs(gradient_y)    # 参数1输入数组，参数alpha乘数因子(默认为1，beta偏移量（0），求绝对值
    cv.imshow('grad_x', gradient_x)
    cv.imshow('grad_y', gradient_y)
    gradient_xy = cv.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    cv.imshow('grad_xy', gradient_xy)


src = cv.imread("F:/Python/image_proceesing/picture/3.png")
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
sobel_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
