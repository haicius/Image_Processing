# Author: 海子
# Create Time: 2022/2/23
# FileName: image_blocking
# Descriptions: 图像尺寸为1080*1920， 缩小1/4后为270*480，然后进行图像分块3*3，90*160/块
import cv2 as cv
import numpy as np
from vidstab import VidStab

orb = cv.ORB_create(150)
fast = cv.FastFeatureDetector_create(100, nonmaxSuppression=True)
star = cv.xfeatures2d.StarDetector_create(maxSize=10)


def image_blocking(frame_gray):  # frame_gray(1080, 1920)
    m1 = cv.getTickCount()
    prev_frame_gray = cv.resize(frame_gray, dsize=None, fx=0.25, fy=0.25)  # prev_frame_gray(270, 480)
    prev_frame_gray1 = prev_frame_gray.copy()
    # print(prev_frame_gray.shape)
    # prev_frame_gray1 = prev_frame_gray.copy()
    # cv.imshow("src", prev_frame_gray)
    h, w = prev_frame_gray.shape[:2]
    block_hs = 3  # 3->(90, 160), 5->(54, 96), 6->(45, 80), 10->(27, 48)
    block_ws = 3  # 3->(90, 160), 5->(54, 96), 6->(45, 80), 10->(27, 48)
    block_h = int(h / block_hs)  # 90
    block_w = int(w / block_ws)  # 96
    # print(block_w)
    prev_kp = np.empty((0, 2), dtype='float32')
    m2 = cv.getTickCount()
    # print("Perframe: time--->" + str((m2 - m1) / cv.getTickFrequency() * 1000) + " ms")
    for i in range(block_hs):
        for j in range(block_ws):
            a1 = cv.getTickCount()
            sub_image = prev_frame_gray[(i * block_h):(i + 1) * block_h, (j * block_w):(j + 1) * block_w]
            # print(sub_image.shape)
            a2 = cv.getTickCount()
            # print("a2Perframe: time--->" + str((a2 - a1) / cv.getTickFrequency() * 1000) + " ms")
            prev_kps = star.detect(sub_image, None)
            prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32')
            a3 = cv.getTickCount()
            # print("a3Perframe: time--->" + str((a3 - a2) / cv.getTickFrequency() * 1000) + " ms")
            print(prev_kps.shape)
            prev_kps[:, 0] = prev_kps[:, 0] + j * block_w
            prev_kps[:, 1] = prev_kps[:, 1] + i * block_h
            # print(prev_kps)
            prev_kp = np.concatenate((prev_kp, prev_kps))
    m2 = cv.getTickCount()
    print("a4Perframe: time--->" + str((m2 - m1) / cv.getTickFrequency() * 1000) + " ms")
    # print("a", prev_kp)
    # print("a", prev_kp.shape)
    for m in prev_kp:
        a, b = m.ravel()
        # print(a, b)
        new_img = cv.circle(prev_frame_gray, (int(a), int(b)), 4, (0, 255, 255), -1)
        cv.imshow("new_image1", new_img)
    return prev_kp


def no_block(frame_gray):
    a1 = cv.getTickCount()
    prev_frame_gray = cv.resize(frame_gray, dsize=None, fx=0.25, fy=0.25)  # prev_frame_gray(270, 480)
    prev_frame_gray1 = prev_frame_gray.copy()
    # print(prev_frame_gray.shape)
    prev_kps = star.detect(prev_frame_gray, None)
    prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32')
    a2 = cv.getTickCount()
    print("no_block_Perframe: time--->" + str((a2 - a1) / cv.getTickFrequency() * 1000) + " ms")
    # print(prev_kps.shape)
    for m in prev_kps:
        a, b = m.ravel()
        # print(a, b)
        new_img = cv.circle(prev_frame_gray1, (int(a), int(b)), 4, (0, 255, 255), -1)
        cv.imshow("new_image2", new_img)
    return prev_kps


# if __name__ == "__main__":


frame_gray = cv.imread("crane.jpg", 0)
t1 = cv.getTickCount()
kp = image_blocking(frame_gray)
kp1 = no_block(frame_gray)
t2 = cv.getTickCount()
# print("Perframe: time--->" + str((t2 - t1) / cv.getTickFrequency() * 1000) + " ms")
print(kp.shape)
print(kp1.shape)
# cv.namedWindow("src", cv.WINDOW_NORMAL)
# cv.namedWindow("blur", cv.WINDOW_NORMAL)
# cv.imshow("src", frame_gray)
# blurred_image = cv.GaussianBlur(frame_gray, (5, 5), 1)  # 对图像进行滤波，增强纹理信息
# cv.imshow("blur", blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()
