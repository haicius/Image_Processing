# Author: 海子
# Create Time: 2021/8/25
# FileName: image_denoise
# Descriptions:  去除图像中噪声的非局部均值去噪算法cv.fastNlMeansDenoising()，cv.fastNlMeansDenoisingColored()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def fastNlMeansDenoisingColored(img):    # 快速非局部均值去噪算法
    # img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    """
    常用参数：

        h:决定滤波器强度的参数。较高的h值可以更好地消除噪点，但同时也可以消除图像细节。（可以设为10）
        hForColorComponents：与h相同，但仅用于彩色图像。
        templateWindowSize：应为奇数。（建议设为7）
        searchWindowSize：应为奇数。（建议设为21）
    """
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(dst)
    # plt.show()
    return dst

def mesh_shift_demo(image):   # 均值漂移算法
    dst = cv.pyrMeanShiftFiltering(image, 0, 15, 15)
    # 第一个参数src，输入图像，8位，三通道的彩色图像，并不要求必须是RGB格式，HSV、YUV等Opencv中的彩色图像格式均可；
    # 第二个参数dst，输出图像，跟输入src有同样的大小和数据格式；
    # 第三个参数sp，定义的漂移物理空间半径大小；
    # 第四个参数sr，定义的漂移色彩空间半径大小；
    # 第五个参数maxLevel，定义金字塔的最大层数；
    # 第六个参数termcrit，定义的漂移迭代终止条件，可以设置为迭代次数满足终止，迭代目标与中心点偏差满足终止，或者两者的结合；
    # 原理：对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，求取该圆形区域内样本的质心，
    # 即密度最大处的点，再以该点为中心继续执行上述迭代过程，直至最终收敛。
    # cv.imshow("mesh_demo", dst)
    return dst


def camera():
    camera = cv.VideoCapture(0)
    width = camera.get(propId=3)
    height = camera.get(propId=4)
    print(width)
    print(height)
    num = 0
    frame_num = 1
    fourcc = cv.VideoWriter_fourcc('X', "V", "I", "D")
    output = cv.VideoWriter('output.avi', fourcc, 20, (480, 640))
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, [480, 640])
        frame = mesh_shift_demo(frame)
        frame = cv.addWeighted(frame, 0.5, frame, 0.5, 50)
        cv.putText(frame, "frame:" + str(frame_num), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv.putText(frame, "second:" + str(round(frame_num / 20, 1)) + "s", (140, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 255), 1)
        cv.imshow("camera_0", frame)

        output.write(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("video_gray", gray)
        c = cv.waitKey(50)
        frame_num += 1
        if c == 32:  # 拍照
            cv.imshow('camera', frame)
            cv.imwrite('frame.png', frame)
            num += 1
            cv.waitKey(0)
            continue
        if c == ord('s'):  # 暂停
            a = cv.waitKey(0)

        if c == 27:  # 退出
            break

    camera.release()
    output.release()


cv.destroyAllWindows()
image = cv.imread('Image/lena.png')
fastNlMeansDenoisingColored(image)
