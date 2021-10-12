# Author: 海子
# Create Time: 2021/7/14
# FileName: edge_detection
# Descriptions:  canny算法：边缘的提取和检测;   直线检测，圆检测;   轮廓检测  ；对象测量（基于轮廓）
import cv2 as cv
import numpy as np


# Canny 算法检测边缘
def edge_demo(image):
    """
         Canny算法步骤：
            第一步，对图像进行高斯模糊，降低噪声，因为梯度运算对噪声很敏感
            第二步，对图像进行灰度变换
            第三步，计算梯度的模和方向
            第四步，对梯度的模应用非极大值抑制，对梯度图像M(x,y)进行细化，确保边缘定位准确   Canny()
            第五步，高低阈值（双阈值和连通性分析）输出二值图像，Th、Tl为高低阈值，高于Th的都被认为时边缘点，赋值255
            小于Tl的都被丢弃0，在Th和Tl之间的像素点进行连通性分析，即判断该点的8邻域是否有大于Th的像素点，如果有，则判定为边缘255
    """
    blurred_image = cv.GaussianBlur(image, (3, 3), 0)
    gray_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)
    gradient_x = cv.Scharr(gray_image, cv.CV_16SC1, 1, 0)
    gradient_y = cv.Scharr(gray_image, cv.CV_16SC1, 0, 1)
    edit_image = cv.Canny(gradient_x, gradient_y,  100, 250)
    cv.imshow("edit_image", edit_image)
    print(edit_image)
    #  为边缘着色（掩码）
    dst = cv.bitwise_and(image, image, mask=edit_image)
    cv.imshow("color image", dst)


# 轮廓的凸包，对于不规则图像，我们期望得到图像最外围的端点并连接构成一个最下的包围框，这个框就是凸包
def convexhull_demo(image):
    dst = cv.bilateralFilter(image, 20, 50, 50)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    t, o = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
    cv.imshow("binary", o)
    contours, hierarchy = cv.findContours(o, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        cv.polylines(image, [hull], True, (0, 0, 255), 2)
    cv.imshow("hull", image)


# 直线检测，原理：霍夫直线变换(Hough Line transform)进行检测，前提是边缘检测已经完成，是平面空间到极坐标空间的转换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    print(lines)
    for line in lines:
        print(type(lines))
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv.imshow("image-lines", image)


def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=150, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_demo", image)


# 圆检测,霍夫圆检测原理：从平面坐标到极坐标转换3个参数C(x0,y0,r),转换到极坐标后，圆心有极大值在反推即可画出圆
# 因为霍夫圆检测对噪声敏感，因此首先要进行中值滤波，第一步检测边缘，发现可能的圆心，第二步从圆心开始计算最佳半径大小
def detect_circles_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=27, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circles", image)


# 轮廓检测：基于图像边缘提取的基础寻找轮廓对象的方法，所以边缘提取的阈值会影响最终轮廓的发现结果
    # 法1，直接基于阈值的二值化图像轮廓提取方法
def contours_demo(image):
    dst = cv.bilateralFilter(image, 20, 50, 50)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    t, o = cv.threshold(gray, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)
    # edges = cv.Canny(gray, 50, 150, apertureSize=3)
    print(t)
    cv.imshow("thresh", o)

    contours, hierarchy = cv.findContours(o, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), -1)
        print(i)
    cv.imshow("detect", image)


# 法2:基于Canny边缘检测的轮廓提取
def contours_demo(image):
    blurred_image = cv.GaussianBlur(image, (3, 3), 0)
    gray_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)
    gradient_x = cv.Sobel(gray_image, cv.CV_16SC1, 1, 0)
    gradient_y = cv.Sobel(gray_image, cv.CV_16SC1, 0, 1)
    edit_image = cv.Canny(gradient_x, gradient_y, 100, 200)
    cv.imshow("edit_image", edit_image)
    contours, hierarchy = cv.findContours(edit_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        print(i)
    cv.imshow("detect", image)


# 基于轮廓特征的对象测量，包括弧长，面积，轮廓拟合， 几何矩计算，单位(像素)
def measure_demo(image):
    dst = cv.bilateralFilter(image, 20, 50, 50)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    t, o = cv.threshold(gray, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)
    cv.imshow("binary", o)
    dst = image.copy()
    contours, hierarchy = cv.findContours(o, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(contours)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)

        x, y, w, h = cv.boundingRect(contour)
        rate = min(w, h) / max(w, h)  # 宽高比
        print("第{0}轮廓, 面积为:{1}， 宽高比：{2}".format(i + 1, area, rate))
        print(x, y, w, h)
        mm = cv.moments(contour)   # 几何矩
        cx = mm["m10"]/mm["m00"]   # 重心
        cy = mm["m01"]/mm["m00"]
        cv.circle(image, (int(round(cx)), int(round(cy))), 2, (0, 255, 255), -1)  # 绘制重心,将坐标近似为整数
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        approxCurve = cv.approxPolyDP(contour, 4, True)  # 近似筛选不同的形状曲线
        # print(approxCurve.shape)
        if approxCurve.shape[0] > 6:    # 圆形
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 4:   # 矩形
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 3:   # 三角形
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)
        if approxCurve.shape[0] > 10:   # 多边形
            cv.drawContours(dst, contours, i, (0, 255, 255), 2)
        cv.imshow("measure-contours", dst)

    cv.imshow("detect", image)


src = cv.imread('F:/Python/image_proceesing/picture/morph02.png')
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('input image', src)
contours_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
