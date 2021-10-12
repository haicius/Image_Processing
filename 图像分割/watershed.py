# Author: 海子
# Create Time: 2021/7/15
# FileName: text
# Descriptions:图像分割，分水岭算法(距离变换，分水岭变换）
# 基于距离的分水岭分割流程：输入图像--灰度变换--二值化--距离变换--寻找种子--生成marker--分水岭变换--输出图像--结束
"""
基本的步骤
通过形态学开运算对原图像O去噪
通过膨胀操作获取“确定背景B”
利用距离变换函数对图像进行运算，并对其进行阈值处理，得到“确定前景F”
计算未知区域UN（UN = O – B – F ）
利用函数connecedComponents对原图像O进行标注
对函数connecedComponents的标注结果进行修正
使用分水岭函数watershed完成对图像的分割
————————————————
版权声明：本文为CSDN博主「duganlx」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40626497/article/details/105270626

"""

import cv2 as cv
import numpy as np


def watershed_demo():
    # remove noise if any
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary-image", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    mb = cv.morphologyEx(mb, cv.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv.dilate(mb, kernel, iterations=1)
    cv.imshow("mor-opt", sure_bg)

    # distance transform
    dist = cv.distanceTransform(sure_bg, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance-t", dist_output*50)

    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)

    surface_fg = np.uint8(surface)
    cv.imshow("surface-bin", surface_fg)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)

    # watershed transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow("result", src)


print("--------- Python OpenCV Tutorial ---------")
src = cv.imread("F:/Python/image_proceesing/picture/coins.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
watershed_demo()
cv.waitKey(0)

cv.destroyAllWindows()



