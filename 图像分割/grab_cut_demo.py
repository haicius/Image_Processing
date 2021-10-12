# Author: 海子
# Create Time: 2021/7/16
# FileName: grab_cut_demo
# Descriptions:   交互式前景提取算法GrabCut:及使用迭代图像分割的交互式前景提取，首先，用户在前景区域周围绘制一个矩形（前景区域应完全
# 位于矩形内部）。然后，算法会对其进行迭代分割，以获得最佳结果。
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('F:/Python/image_proceesing/picture/5.jpg')
img = cv.resize(img, (512, 700))

mask = np.zeros(img.shape[:2], np.uint8)    # 创建一个和原始图像相同大小掩码
cv.imwrite('F:/Python/image_proceesing/picture/newmark2.png', mask)
bgdModel = np.zeros((1, 65), np.float64)    # 算法内部使用的数组
fgdModel = np.zeros((1, 65), np.float64)   # 算法内部使用的数组
"""
cv.rectangle(img, (50, 50), (450, 650), (0, 0, 255), 2)
cv.imshow("src", img)
cv.waitKey()
cv.destroyAllWindows()
"""
rect = (50, 50, 450, 650)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 1. np.where(condition, x, y)，  满足条件(condition)，输出x，不满足输出y。
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

img = img * mask2[:, :, np.newaxis]
plt.imshow(img), plt.colorbar(), plt.show()
