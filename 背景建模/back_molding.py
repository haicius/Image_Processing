# Author: 海子
# Create Time: 2021/8/3
# FileName: back_molding
# Descriptions:  背景建模
"""
背景建模：常用于处理对目标检测、提取，跟踪的图像处理任务，背景像素设置为0， 目标为1
方法一： 帧差法,依次对时间上连续的两帧图像进行差分运算，不同帧图像做对应元素运算，判断灰度差的绝对值，若绝对值大于设定的阈值，
        即可判断为运动目标
        问题：空洞，噪音
方法二： 高斯混合模型GMM，在进行前景检测前，先对背景进行训练，对图像中每个背景采用一个混合高斯模型进行模拟，每个背景的混合高斯的个数可以自
        适应。然后在测试阶段，对新来的像素进行GMM（高斯混合模型）匹配，如果该像素值能够匹配其中一个高斯，则认为是背景，否则认为是前景。
        由于整个过程GMM模型在不断更新学习中，所以对动态背景有一定的鲁棒性。最后通过对一个有树枝摇摆的动态背景进行前景检测，取得了较好的效果。
        在视频中对于像素点的变化情况应当是符合高斯分布，背景的实际分布应当是多个高斯分布混合在一起，每个高斯模型也可以带有权重
        混合高斯模型学习方法
        高斯模型学习方法
            1.首先初始化每个高斯模型矩阵参数。
            2.取视频中T帧数据图像用来训练高斯混合模型。来了第一个像素之后用它来当做第一个高斯分布。
            3.当后面来的像素值时，与前面已有的高斯的均值比较，
            4、如果该像素点的值与其模型均值差在3倍的方差内，则属于该分布，并对其进行参数更新。
        混合高斯模型测试方法
            在测试阶段，对新来像素点的值与混合高斯模型中的每一个均值进行比较，如果其差值在2倍的方差之间的话，则认为是背景，否则认为是前景。
            将前景赋值为255，背景赋值为0。这样就形成了一副前景二值图。
"""
import numpy as np
import cv2

# 经典的测试视频
cap = cv2.VideoCapture('test.avi')
if not cap.isOpened():
    print("视频打开失败")
    exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(propId=3)
width = cap.get(propId=4)
print(fps)
print(width)
print(height)

# 实例化一个视频保存对象
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
# fourcc = cv2.VideoWriter_fourcc(*"MPEG")
out_contours = cv2.VideoWriter("out_contours.avi", fourcc, 20, (576, 768))
out_fg = cv2.VideoWriter("out_fg.avi", fourcc, 20, (576, 768))

# 形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 创建一个结构单元

# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)  # 应用到每一帧中提取背景,包括计算前景掩码和更新背景
    # 形态学开运算去噪点
    fg_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CROSS, kernel, iterations=1)
    fg_mask1 = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # 寻找视频中的轮廓
    image, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c_t = []
    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        if perimeter > 100:
            # 找到一个直矩形（不会旋转）
            # x, y, w, h = cv2.boundingRect(c)
            # # 画出这个矩形
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            c_t.append(c)
    cv2.drawContours(frame, c_t, -1, (0, 0, 255), 2)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('frame', frame)
    cv2.imshow('fg_mask1', fg_mask1)
    out_contours.write(frame)
    out_fg.write(fg_mask1)
    k = cv2.waitKey(60)
    if k == 27:
        break

cap.release()
out_fg.release()
cv2.destroyAllWindows()
