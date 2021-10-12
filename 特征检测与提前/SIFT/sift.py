# Author: 海子
# Create Time: 2021/8/10
# FileName: sift
# Descriptions:  尺度不变特征变换sift
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/01.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create(202)
t1 = cv.getTickCount()
kp, des = sift.detectAndCompute(gray, None)
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("关键点检测时间消耗：%s" % (time*1000))
print("kp:", len(kp))
img = cv.drawKeypoints(gray, kp, img,
                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.title("sift")
plt.show()
cv.imwrite('sift_keypoints.jpg', img)

