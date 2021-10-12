# Author: 海子
# Create Time: 2021/8/19
# FileName: BRIEF
# Descriptions:  特征生成流程：特征点快速提取法CenSurf（STAR）+BRIEF特征描述+汉明距离匹配
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/01.jpeg', 0)

# 初始化Star检测器
star = cv.xfeatures2d.StarDetector_create()

# 初始化BRIEF提取器
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
t1 = cv.getTickCount()
# 找到STAR的关键点
kp = star.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
# 打印所有默认参数
print("SURF-->time=%s ms" % (time*1000))
cv.imshow("kp", img2)
# 计算BRIEF的描述符
kp, des = brief.compute(img, kp)  # 该des可以是128、256或512。OpenCV支持所有这些，但默认情况下将是为单位表示，因此值将为16、32和64）。因此，一旦获得此信息，就可以些描述符。

print(brief.descriptorSize())
print(des.shape)
cv.waitKey()
cv.destroyAllWindows()
