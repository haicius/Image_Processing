# Author: 海子
# Create Time: 2021/7/15
# FileName: face_detect
# Descriptions:人脸检测(人脸跟踪), 眼睛特效：先使用cv.CascadeClassifiter(filename),创建一个级联分类器对象cascade，然后使用
# 创建好的分类器调用cascade.detectMultiScale(image)对图像进行识别，返回的对象是一个目标区域的数组
import cv2 as cv
import numpy as np


def face_cascade(image):   # 人脸检测
    face_cascade = cv.CascadeClassifier("E:/python3.8.1/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(image, 1.15, 3)
    # for (x, y, w, h) in faces:
    #     cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return faces


def overlay_img(img, img_over, img_over_x, img_over_y):         # 覆盖图像
    """
    覆盖图像
    :param img: 背景图像
    :param img_over: 覆盖的图像
    :param img_over_x: 覆盖图像在背景图像上的横坐标
    :param img_over_y: 覆盖图像在背景图像上的纵坐标
    :return: 两张图像合并之后的图像
    """
    img_h, img_w, img_p = img.shape  # 背景图像宽、高、通道数
    img_over_h, img_over_w, img_over_c = img_over.shape  # 覆盖图像高、宽、通道数
    if img_over_c == 3:  # 通道数小于等于3
        img_over = cv.cvtColor(img_over, cv.COLOR_BGR2BGRA)  # 转换成4通道图像
    for w in range(0, img_over_w):  # 遍历列
        for h in range(0, img_over_h):  # 遍历行
            if img_over[h, w, 3] != 0:  # 如果不是全透明的像素
                for c in range(0, 3):  # 遍历三个通道
                    x = img_over_x + w  # 覆盖像素的横坐标
                    y = img_over_y + h  # 覆盖像素的纵坐标
                    if x >= img_w or y >= img_h:  # 如果坐标超出最大宽高
                        break  # 不做操作
                    img[y, x, c] = img_over[h, w, c]  # 覆盖像素
    return img  # 完成覆盖的图像


def eye_tree_demo():   # 眼睛特效
    while camera.isOpened():
        ret, frame = camera.read()  # ret返回一个布尔值，验证是否读取视频；frame表示每一帧图像，RGB格式
        if not ret:  # 判断摄像头是否正常
            break
        frame = cv.flip(frame, 1)  # 将画面左右翻转    # 人脸视频帧
        # frame = cv.bilateralFilter(frame, 10, 50, 50)
        frame = cv.pyrMeanShiftFiltering(frame, 0, 15, 15)
        frame = cv.addWeighted(frame, 0.5, frame, 0.5, 50)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化算子，立体感强，边缘细节，图像增强
        # 卷积核算子很重要，不同的算子和图形卷积有不同的效
        frame = cv.filter2D(frame, -1, kernel=kernel)
        glass_img = cv.imread("E:/python/project/B_jiazhigang/picture/glass.png")  # 加载眼睛特效
        height, width, channels = glass_img.shape
        faces = face_cascade(frame)
        for (x, y, w, h) in faces:  # 遍历所有人脸的区域
            gw = w  # 眼镜缩放之后的宽度
            gh = int(height * w / width)  # 眼镜缩放之后的高度度
            glass_img = cv.resize(glass_img, (gw, gh))  # 按照人脸大小缩放眼镜
            cv.imshow('glass', glass_img)
            overlay_img(frame, glass_img, x, y + int(h * 1 / 3))  # 将眼镜绘制到人脸上
        cv.imshow('camera', frame)
        c = cv.waitKey(10)
        if c == 27:
            break
        if c == 32:  # 拍照
            cv.imshow('camera', frame)
            cv.imwrite('picture/frame.png', frame)
            cv.waitKey(0)
            continue


def face_detect():   # 视频人脸跟踪
    while camera.isOpened():
        ret, frame = camera.read()  # ret返回一个布尔值，验证是否读取视频；frame表示每一帧图像，RGB格式
        if not ret:  # 判断摄像头是否正常
            break
        frame = cv.flip(frame, 1)  # 将画面左右翻转
        face_cascade(frame)
        cv.imshow('camera', frame)
        c = cv.waitKey(1)
        if c == 27:
            break
        # if c == ord('e'):
        #     break


print('-------Hi python----------')
camera = cv.VideoCapture(0, cv.CAP_DSHOW)  # 参数0代表计算机第一个摄像头

eye_tree_demo()
camera.release()
cv.destroyAllWindows()  # 释放所有内存
print('HI,python')
