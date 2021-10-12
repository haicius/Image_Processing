# Author: 海子
# Create Time: 2021/7/9
# FileName: gray_hist
# Descriptions:绘制直方图，应用于图像二值化的阈值选择，直方图均衡化(图像增强)，直方图投影(目标识别与跟踪)
# dims:需要统计的特征数目，bins:每个特征空间参与的参数， range:每个特征空间的取值范围
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 使用pyplot直接绘制直方图
def pyplot_hist(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


# 使用cv.calcHist()直接绘制直方图
def cv_hist(image):
    # calcHist();参数包括原始图像，通道数，掩码，bins,范围。。。
    # hist_b = cv.calcHist([image], [0], None, [256], [0, 255])
    # hist_g = cv.calcHist([image], [1], None, [256], [0, 255])
    # hist_r = cv.calcHist([image], [2], None, [256], [0, 255])
    # hist = hist_b + hist_g + hist_r
    # plt.subplot(2, 1, 1)
    # plt.plot(hist_b, 'b'), plt.axis("off")
    # plt.plot(hist_g, 'g')
    # plt.plot(hist_r, 'r')
    # plt.plot(hist, 'y')
    # plt.show()
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):  # 一个索引序列
        hist = cv.calcHist([image], [i], None, [256], [0, 255])
        plt.plot(hist, color=color)
        plt.xlim(0, 256)
    plt.show()


# 掩码运算后绘制直方图，逻辑与运算
def mask(image):
    mask = np.zeros(image.shape, np.uint8)
    mask[50:300, 50:300] = 255
    mask_image = cv.bitwise_and(image, mask)  # 参数原图像以及自定义的掩码
    cv.imshow('mask', mask_image)
    mask_image = plt.hist(mask_image.ravel(), 256)
    plt.show()


# 直方图全局均衡化,提高图像对比度,必须是灰度图，可提高图像的对比度，图像增强
def equalize_hist(image):
    pyplot_hist(image)
    equal_image = cv.equalizeHist(image)
    cv.imshow("equalize_hist", equal_image)


#  自适应局部直方图均衡化，
# 原理：对在图像像素个数多的灰度值进行展宽，对像素个数少的进行归并
#  1、计算直方图 2、归一化 3、计算每种灰度的累计分布 4、进行直方图均衡化计算
def adaptive_equ_hist(image):
    pyplot_hist(image)
    clahe_image = cv.createCLAHE(clipLimit=400.0, tileGridSize=(8, 8))
    dst = clahe_image.apply(image)
    cv.imshow("clahe_hist", dst)


# 直方图比较,相关性检查
def create_rgb_hist(image):
    h, w, c = image.shape
    rgb_hist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize) * 16*16+np.int(g/bsize)*16 + np.int(r/bsize)
            rgb_hist[np.int(index), 0] = rgb_hist[np.int(index), 0] + 1
    return rgb_hist


'''
直方图四种比较方法：cv.compareHist()，可用于目标检测
1、相关性检查：越接近1越相似，越接近0越不相似
2、直方图相交：值越大越相交
3、卡方：越大越不相似
4、巴氏距离：越大越不相似


'''
def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：%s ，相关性：%s， 卡方：%s"%(match1, match2, match3))


# 直方图反向投影，HSV和RGB色彩空间，适用于目标跟踪，颜色识别；
# 反向投影是反映直方图模型在目标图像中的分布情况；
# 简单点说就是用直方图模型去目标图像中寻找是否有相似的对象。通常用HSV色彩空间的HS两个通道直方图模型
def back_projection_demo():
    sample = cv.imread("F:/Python/image_proceesing/picture/5.1.jpg")
    target = cv.imread("F:/Python/image_proceesing/picture/5.jpg")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow('sample', sample)
    cv.imshow('target', target)

    roi_hist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 255])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)
    cv.imshow("backProjection", dst)


def hist2D_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D_HIST")
    plt.show()


image = cv.imread("F:/Python/image_proceesing/picture/3.png")
src = cv.imread("F:/Python/image_proceesing/picture/4.png")
# cv.imshow("a", image)
# cv.imshow("b", src)

# print(image.shape)
# image1 = image.copy()
# pyplot_hist(image1)  # 灰度直方图

# image2 = image.copy()
# cv_hist(image2)

# image3 = image.copy()
# image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
# image4 = image.copy()
# mask(image4)  # 掩码后计算直方图

# image5 = image.copy()
# equalize_hist(image5)  # 直方图均衡化

# image6 = image.copy()
# adaptive_equ_hist(image6)  #

# hist_compare(image, src)

# hist2D_demo(image)
back_projection_demo()
cv.waitKey(0)
cv.destroyAllWindows()
