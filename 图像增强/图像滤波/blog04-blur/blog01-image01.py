# -*- coding:utf-8 -*-
import cv2
import numpy as np

# 读取图片
img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, chn = img.shape

# 加椒盐噪声
for i in range(5000):
    x = np.random.randint(0, rows)
    y = np.random.randint(0, cols)
    img[x, y, :] = 255


# 添加高斯噪声
def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gauss_noise_demo(image):   # 添加高斯噪声，均值为0，方差为sigam的平方
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.randint(0, 50, size=3)
            """ 
             # 返回一个随机数组，其范围【0，50】,1*3的数组
            """
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("gauss_noise", image)


cv2.imshow("noise", img)

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
