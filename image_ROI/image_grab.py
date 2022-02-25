# Author: 海子
# Create Time: 2021/11/28
# FileName: image_grab
# Descriptions:
import cv2 as cv
import numpy as np

img = cv.imread("../picture/0001.jpg")
roi = cv.selectROI(windowName="roi", img=img, showCrosshair=True, fromCenter=False)
x, y, w, h = roi
x1 = x
y1 = y
x2 = x + w
y2 = y + h
print(x, y, w, h)
cv.imshow("draw_roi", roi)
image_roi1 = img[y1:y2, x1:x2]
cv.imshow("image_roi1", image_roi1)
mask = np.zeros(img.shape, np.uint8)
mask[y1:y2, x1:x2] = 255
cv.imshow("mask", mask)
image_roi2 = cv.bitwise_and(img, mask)
cv.imshow("image_roi2", image_roi2)

cv.waitKey(0)
cv.destroyAllWindows()
