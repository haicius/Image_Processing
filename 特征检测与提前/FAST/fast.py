# Author: 海子
# Create Time: 2021/8/18
# FileName: fast
# Descriptions: FAST 角点检测
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/01.jpeg', 0)
# 用默认值初始化FAST对象

t1 = cv.getTickCount()
fast = cv.FastFeatureDetector_create(80)
# 寻找并绘制关键点
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
# 打印所有默认参数
print("SURF-->time=%s ms" % (time*1000))
cv.imshow("KP-NonMax", img2)
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
cv.imwrite('fast_true.png', img2)
# 关闭非极大抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imshow("KP-NOnonMax", img3)
cv.imwrite('fast_false.png', img3)
cv.waitKey(0)
cv.destroyAllWindows()
