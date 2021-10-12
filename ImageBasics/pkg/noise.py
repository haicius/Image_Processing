# Author: 海子
# Create Time: 2021/8/3
# FileName: noise
# Descriptions:  添加图像噪声，椒盐噪声，高斯噪声
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                output[i, j, :] = 0
            if rdn > thres:
                output[i, j, :] = 255
            else:
                output[i, j] = image[i, j]
    return output


def clamp(pv):
    """防止图像添加高斯噪声后溢出"""
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gaussian_noise_demo(image):
    """添加高斯噪声"""
    h, w, c = image.shape
    for row in range(0, h):
        for col in range(0, w):
            s = np.random.randint(0, 65)  # 产生随机数，每次产生三个
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s)
            image[row, col, 1] = clamp(g + s)
            image[row, col, 2] = clamp(r + s)
    return image


a = 10
src = cv.imread("Image/1.jpg", cv.IMREAD_COLOR)
cv.imshow("input_image", src)

sp_noise = sp_noise(src, 0.01)
cv.imshow("sdd_sp_noise", sp_noise)
cv.imwrite("Image/sp.noise.jpg", sp_noise)

image = gaussian_noise_demo(src)
cv.imshow("add_gauss_noise", image)
cv.imwrite("Image/gauss.noise.jpg", image)
cv.waitKey(0)
cv.destroyAllWindows()
