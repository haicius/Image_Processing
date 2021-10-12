# Author: 海子
# Create Time: 2021/8/19
# FileName: Brute_Force
# Descriptions: 暴力匹配,包含两个方法，一对一匹配，Knn匹配

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('img/box.png', 0)
img2 = cv.imread('img/box_in_scene.png', 0)


def orb_bf_demo():
    # 创建ORB对象
    t1 = cv.getTickCount()
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    print(des1.shape, des2.shape)

    # 创建bf对象，并设定初始值
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    print(matches)
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    print("ORB_BF-->%s ms" % (time * 1000))
    # 将匹配结果按特征点之间的距离进行降序排列
    matches = sorted(matches, key=lambda x: x.distance)

    # 前10个匹配
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

    cv.namedWindow('orb_img', cv.WINDOW_AUTOSIZE)
    cv.imshow('orb_img', img3)


def sift_bf_demo():
    # 创建SIFT对象
    t1 = cv.getTickCount()
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用默认参数 cv.Norm_L2 ,crossCheck=False
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    print("SIFT_BF-->%s ms" % (time * 1000))
    # 比值测试，首先获取与A距离最近的点B（最近）和C（次近），只有当B/C小于阈值时（0.75）才被认为是匹配
    # 因为假设匹配是一一对应的，真正的匹配的理想距离是0
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv.namedWindow('sift_img', cv.WINDOW_AUTOSIZE)
    cv.imshow('sift_img', img3)


orb_bf_demo()
sift_bf_demo()
cv.waitKey(0)
cv.destroyAllWindows()
