# Author: 海子
# Create Time: 2021/7/30
# FileName: HarrisCornerDetection
# Descriptions: cv.cornerHarris()的使用

import cv2 as cv
import numpy as np

src = cv.imread("img/03.jpeg")
print("src.shape:", src.shape)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
t1 = cv.getTickCount()
dst = cv.cornerHarris(gray, 2, 3, 0.04)  # 返回值是一个与原图像相同大小的灰度图像
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
print(dst)
dst = np.uint8(dst)
# 寻找质心
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# 定义停止和完善拐角的条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
# 绘制
res = np.hstack((centroids, corners))
res = np.int0(res)
src[res[:, 1], res[:, 0]] = [0, 0, 255]
src[res[:, 3], res[:, 2]] = [0, 255, 0]
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("关键点检测时间消耗：%s" % (time*1000))

"""
cv.cornerHarris(gray, 5, 5, 0.04) :
src参数1：数据类型为float32的灰度图像
blocksize参数2；角点检测中指定区域的范围
ksize参数3：梯度算子空间大小
k值：0.04——0.06  ,敏感因子，越小越容易检测角点
"""

print("dst.shape:", src.shape, src.dtype)
cv.imshow("corner.detection", src)
cv.waitKey()
cv.destroyAllWindows()
