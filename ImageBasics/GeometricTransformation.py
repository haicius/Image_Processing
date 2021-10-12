# Author: 海子
# Create Time: 2021/8/3
# FileName: GeometricTransformation
# Descriptions: 图像几何变换，平移，旋转，缩放， 剪切， 镜像
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


def resize_demo(image):
    """
    图像缩放
    """
    image1 = cv.resize(image, (200, 200))
    image2 = cv.resize(image, dsize=None, fx=0.5, fy=0.5)
    image3 = cv.pyrDown(image)
    image4 = cv.pyrUp(image3)
    cv.imshow("resize_demo", image4)


"""
图像镜像：np.fliplr(src), cv.flip(src, 1),  
上下翻转：np.flipud()src
"""

def warp_affine_demo(image):
    """
    仿射变换, 平移， 透视， 旋转。。。，可用于图像矫正
    """
    image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
    M1 = np.float32([[1, 0, 100], [0, 1, 100]])   # 平移矩阵
    move_image = cv.warpAffine(image, M1, (image.shape[1], image.shape[0]))    # 　映射函数（平移）

    M2 = cv.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), 30, 2)  # 旋转矩阵
    rotion_image = cv.warpAffine(image, M2, (image.shape[1], image.shape[0]))    # 　映射函数（旋转）

    pos1 = np.float32([[50, 50], [200, 50], [20, 200]])
    pos2 = np.float32([[100, 100], [200, 50], [100, 250]])
    M3 = cv.getAffineTransform(pos1, pos2)
    affine_image = cv.warpAffine(image, M3, (image.shape[1], image.shape[0]))   # 映射函数（仿射）

    flip_image = cv.flip(image, 1)   # 翻转

    pos3 = np.float32([[114, 82], [287, 156], [8, 320], [216, 333]])
    pos4 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
    M4 = cv.getPerspectiveTransform(pos3, pos4)
    perspective_image = cv.warpPerspective(image, M4, (image.shape[1], image.shape[0]))
    titles = ["image", "move_image", "flip_image", "rotation", "affine_image", "perspective"]
    images = [image, move_image, flip_image, rotion_image, affine_image, perspective_image]
    for i in range(len(images)):
        plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def image_correction():
    """
    图像矫正
    """
    src = cv.imread("Image/paper.png", 1)
    cv.imshow("input", src)
    r, c = src.shape[:2]
    img = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 250, apertureSize=3)
    cv.imshow("canny", edges)
    cv.imwrite("Image/canny.jpg", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=90, maxLineGap=10)
    for x1, y1, x2, y2 in lines[0]:
        print((x1, y1), (x2, y2))
    for x1, y1, x2, y2 in lines[1]:
        print((x1, y1), (x2, y2))
    for x1, y1, x2, y2 in lines[0]:
        cv.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 1)
    pos3 = np.float32([[114, 82], [287, 156], [8, 320], [216, 333]])
    pos4 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
    M4 = cv.getPerspectiveTransform(pos3, pos4)
    image_correction = cv.warpPerspective(src, M4, (188, 280))
    cv.imshow("correct", image_correction)
    cv.imwrite("Image/correct.jpg", image_correction)


src = cv.imread('Image/lena.png', 1)
img1 = cv.imread('Image/gauss.noise.jpg', 0)
img2 = cv.imread('Image/sp.noise.jpg', 1)
# cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
# cv.imshow('input image', src)

t1 = cv.getTickCount()  # 时钟函数，单位s
image_correction()
t2 = cv.getTickCount()

time = (t2 - t1) / cv.getTickFrequency()  # 计算图片转化时间
print('time-->%s ms' % (time * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
