# Author: 海子
# Create Time: 2021/7/15
# FileName: Video_process
# Descriptions:
# Author: 海子
# Create Time: 2021/7/7
# FileName: test2
# Descriptions: 视频操作
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


"""
摄像头操作
功能一：读取电脑的一个摄像头并显示320*480
功能二；按空格键保存当前照片并显示某一时刻的图像，拍照功能
功能三：按esc退出摄像头，并保存视频
功能四：将摄像头视频转换其他色彩空间
功能五：在视频画面上显示时长，frames，fps,width,height,帧数等属性
功能五：美颜，磨皮，
功能六：人脸跟踪，。。。
"""
camera = cv.VideoCapture(0)
height = camera.get(propId=3)
width = camera.get(propId=4)
print(height)
print(width)
num = 0
frame_num = 1
fourcc = cv.VideoWriter_fourcc('X', "V", "I", "D")
output = cv.VideoWriter('output.avi', fourcc, 20, (320, 480))
while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (320, 480))
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
        cv.imwrite('picture/frame.png', frame)
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

