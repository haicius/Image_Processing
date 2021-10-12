# Author: 海子
# Create Time: 2021/6/26
# FileName: tutorial_3
# Descriptions:色彩空间
import cv2 as cv
import numpy as np

capture = cv.VideoCapture('E:/python/project/B_jiazhigang/picture/red_cap.mp4')
fourcc = cv.VideoWriter_fourcc(*"XVID")
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
print(height)
print(width)
fps = capture.get(cv.CAP_PROP_FPS)

out = cv.VideoWriter('E:/python/project/B_jiazhigang/picture/HSV_bitwise_and.avi', fourcc, fps, (480, 640))


def extrace_video_demo():  # 视频信息
    # camera = cv.VideoCapture(0, cv.CAP_DSHOW)  # 参数0代表计算机第一个摄像头
    while True:
        ret, frame = capture.read()  # ret返回一个布尔值，验证是否读取视频；frame表示每一帧图像，RGB格式
        if not ret:
            break
        # frame = cv.flip(frame, 1)  # 将画面左右翻转
        frame = cv.resize(frame, (480, 640))
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # HSV很重要，H(0:180),S(0:255),V(0:255)
        lower_hsv = np.array([156, 43, 46])     # 红色HSV低值
        high_hsv = np.array([180, 255, 255])    # 红色HSV高值
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)    # 掩码
        dst = cv.bitwise_and(frame, frame, mask=mask)   # 位和
        out.write(dst)
        cv.imshow('redcap', frame)
        cv.imshow('HSV', dst)
        c = cv.waitKey(1)
        if c == 32:
            cv.waitKey(0)
            continue

        if c == 27:
            capture.release()
            break


def save_video():
    pass


def image_color_space():  # 色彩空间
    rgb = cv.imread('E:/python/project/B_jiazhigang/picture/lena.png')
    cv.namedWindow('RGB', cv.WINDOW_AUTOSIZE)
    cv.imshow('RGB', rgb)

    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    cv.imshow('GRAY', gray)
    cv.imwrite('E:/python/project/B_jiazhigang/picture/gray.png', gray)

    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)  # HSV很重要，H(0:180),S(0:255),V(0:255)
    cv.imshow('HSV', hsv)
    cv.imwrite('E:/python/project/B_jiazhigang/picture/HSV.png', hsv)

    ycrcb = cv.cvtColor(rgb, cv.COLOR_BGR2YCrCb)
    cv.imshow('YCrCb', ycrcb)
    cv.imwrite('E:/python/project/B_jiazhigang/picture/YCrCB.png', ycrcb)

    yuv = cv.cvtColor(rgb, cv.COLOR_BGR2YUV)
    cv.imshow('YUV', yuv)
    cv.imwrite('E:/python/project/B_jiazhigang/picture/YUV.png', yuv)


def rgb_split():     # 图像通道分离与合并
    rgb = cv.imread('E:/python/project/B_jiazhigang/picture/lena.png')
    cv.namedWindow('RGB', cv.WINDOW_AUTOSIZE)
    cv.imshow('RGB', rgb)
    b, g, r = cv.split(rgb)
    cv.imshow('blue', b)
    cv.imshow('green', g)
    cv.imshow('red', r)
    rgb[:, :, 2] = 0
    cv.imshow('changed_red', rgb)
    src = cv.merge([b, g, r])
    cv.imshow('merge_RGB', src)


t1 = cv.getTickCount()  # 时钟函数
# image_color_space()
extrace_video_demo()

# rgb_split()
t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()  # 计算图片转化时间
print('time-->%s ms' % (time * 1000))
# cv.waitKey(0)
capture.release()
out.release()
cv.destroyAllWindows()  # 释放所有内存
print('HI,python')
