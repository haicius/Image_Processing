# Author: 海子
# Create Time: 2021/8/19
# FileName: orb
# Descriptions:  ORB 改进Fast特征点检测+改进BRIEF特征描述
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/05.jpeg', 0)
t1 = cv.getTickCount()
# 初始化ORB检测器
orb = cv.ORB_create(500)

# 用ORB寻找关键点
kp = orb.detect(img, None)
print(len(kp))
# 用ORB计算描述符
t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()
print("ORB-->%s ms" % (time * 1000))
kp, des = orb.compute(img, kp)
print(des.shape)
# 仅绘制关键点的位置，而不绘制大小和方向

img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()
