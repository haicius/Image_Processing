# Author: 海子
# Create Time: 2021/7/9
# FileName: thresh_gauss_noise
# Descriptions:(基于平滑滤波后阈值，基于边缘后阈值)阈值处理--->主要对所提取的目标图像进行形态分析（腐蚀和膨胀）
# ，开运算，闭运算，顶帽，形态学操作（如周长，面积，圆形度等）,主要是对二值图像进行处理
import cv2 as cv
import numpy as np


# 阈值处理函数：cv.threshhold()函数,全局阈值处理或单阈值处理的方法
def binary_thresh(image):
    t1, dst1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    t2, dst2 = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
    t3, dst3 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO)
    t4, dst4 = cv.threshold(image, 100, 255, cv.THRESH_TOZERO_INV)
    t5, dst5 = cv.threshold(image, 127, 255, cv.THRESH_TRUNC)

    cv.imshow("binary", dst1)
    cv.imshow("binary_inv", dst2)
    cv.imshow("tozero", dst3)
    cv.imshow("tozero_inv", dst4)
    cv.imshow("trunc", dst5)


# cv.adaptivethreshhold():自适应阈值处理，也是区域或局部阈值处理，多阈值处理
def adaptive_thresh(image):
    # image必须是灰度图像
    dst1 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,  3, 3)
    # 对局部区域平均所有像素平均加权， 局部区域5*5大小， 阈值等于均值减去常数3
    dst2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 3)
    # 对局部区域所有像素高斯平均加权
    cv.imshow("mean_thresh", dst1)
    cv.imshow("gauss_thresh", dst2)


# 自定义阈值二值化
def custom_threshed(image):
    h, w = image.shape
    m = np.reshape(image, [1, w*h])
    mean = m.sum() / (w*h)
    print('mean: ', mean)
    binary = cv.threshold(image, mean, 255, cv.THRESH_TRIANGLE)
    cv.imshow('binary', image)


# 超大图像阈值二值化，分块原理
def big_image_threshed(image):
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    for row in range(0, h, ch):
        for col in range(0, h, cw):
            roi = image[row:row+ch, col:col+cw]
            t, dst = cv.threshold(roi, 0, 255, cv.THRESH_TRIANGLE | cv.THRESH_OTSU)  # 可改为局部阈值（效果好）
            image[row:row + ch, col:col + cw]
            print(np.std(dst), np.mean(dst))
    cv.imwrite("picture/", image)


# 最大化类间方差或三角法(适用于单峰直方图)求全局最优阈值
def ostu_thresh(image):
    t,  o = cv.threshold(image, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY_C)
    # t,  o = cv.threshold(image, 0, 255, cv.THRESH_TRIANGLE+cv.THRESH_BINARY_C)
    print(t)
    cv.imshow('thresh', o)
    return o


# 自定义一个结构元素
def custom_element_demo():
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return kernel


# 形态学腐蚀，适用于去除小颗粒噪声，以及消除目标物之间的粘连，可以抹除一些目标外部的细节
def erode_image(image):
    kernel = custom_element_demo()
    # kernel = np.ones((5, 5), np.uint8)  # 结构元素为某一正方形确定形状的结构元素
    erode = cv.erode(image, kernel, iterations=1)
    cv.imshow('erode_image', erode)
    return erode


# 膨胀，适用于填补目标区域中的空洞以及消除目标区域内的小颗粒噪声，将会放大一些外部细节
def dilate_image(image):
    kernel = np.ones((5, 5), np.uint8)  # 结构元素为某一确定形状的结构元素
    dilate = cv.dilate(image, kernel, iterations=1)
    cv.imshow('dilate_image', dilate)
    return dilate


# 开运算，闭运算，顶帽，黑帽
def open_close_demo(image):
    # 开运算，先腐蚀后膨胀，主要用于除去一些外部细节(或噪声），同时保持目标像素基本保持不变
    # 闭运算，像膨胀后腐蚀，主要用于出去一些内部细节(或噪声）
    kernel = custom_element_demo()
    erode = erode_image(image)
    ero_dilate = cv.dilate(erode, kernel=kernel, iterations=1)
    cv.imshow("erode_dilate", ero_dilate)


# 形态学方法API，封装了一些常见运算，腐蚀，膨胀，开闭，梯度，顶帽，黑帽
# 注意：不同形状的结构元素可以从图像得到不同的效果，重点是选择一个合适的结构元素
# 梯度：基本梯度(dialate-erode),内部梯度(src-erode),外部梯度(dialate-src)
def morphology_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    dst1 = cv.morphologyEx(image, cv.MORPH_ERODE, kernel=kernel, iterations=1)
    dst2 = cv.morphologyEx(image, cv.MORPH_DILATE, kernel=kernel)
    dst3 = cv.morphologyEx(image, cv.MORPH_OPEN, kernel=kernel)
    dst4 = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel=kernel)
    dst5 = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel=kernel, iterations=1)
    dst6 = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel=kernel)
    dst7 = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel=kernel)
    cv.imshow("dst", dst4)


image = cv.imread("F:/Python/image_proceesing/picture/morph01.png", 0)
# binary_thresh(image)
# adaptive_thresh(image)
# ostu_thresh(image)
# custom_threshed(image)
# big_image_threshed(image):
t, o = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(t)
cv.imshow("binary", o)
# erode_image(o)
# dilate_image(o)
# open_close_demo()
morphology_demo(o)
cv.waitKey(0)
cv.destroyAllWindows()
