# Author: 海子
# Create Time: 2021/8/22
# FileName: camera_1
# Descriptions: 相机校准,第一步:寻找拐角；第二步：校准
import numpy as np
import cv2 as cv
import glob

# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# 用于存储所有图像的对象点和图像点的数组。
objpoints = []  # 真实世界中的3d点
imgpoints = []  # 图像中的2d点
images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 找到棋盘角落
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # 如果找到，添加对象点，图像点（细化之后）
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # 绘制并显示拐角
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()
