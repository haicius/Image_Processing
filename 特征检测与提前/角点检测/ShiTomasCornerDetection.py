# Author: 海子
# Create Time: 2021/8/18
# FileName: ShiTomasCornerDetection
# Descriptions:  Shi-Tomasi拐角检测器

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/03.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
t1 = cv.getTickCount()
corners = cv.goodFeaturesToTrack(gray, 0, 0.01, 8)
print(corners.shape)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("关键点检测时间消耗：%s" % (time*1000))
plt.imshow(img), plt.show()
