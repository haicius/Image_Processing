# Author: 海子
# Create Time: 2021/11/28
# FileName: video_ROI
# Descriptions:
import os
import cv2
import numpy as np

cap = cv2.VideoCapture("001.mp4")

print("FPS--->", (cap.get(cv2.CAP_PROP_FPS)))
print("Height--->", (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Width--->", (cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
ret, first_image = cap.read()
first_image = cv2.resize(first_image, dsize=None, fx=0.5, fy=0.5)
print(first_image.shape)
roi = cv2.selectROI(windowName="please select ROI, Then Enter Exit", img=first_image, showCrosshair=True,
                    fromCenter=False)
x, y, w, h = roi
x1 = x
y1 = y
x2 = x + w
y2 = y + h
print(x, y, w, h)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    if ret == False:
        break
    mask = np.zeros(img.shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    image_roi = cv2.bitwise_and(img, mask)
    cv2.imshow("ROI", image_roi)
    c = cv2.waitKey(20)
    if c == 27:  # esc退出
        break
    if c == 32:  # 空格暂停
        cv2.waitKey(0)
        continue

cap.release()
cv2.destroyAllWindows()
