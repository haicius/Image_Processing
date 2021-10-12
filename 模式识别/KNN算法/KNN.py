# Author: 海子
# Create Time: 2021/8/23
# FileName: KNN
# Descriptions: k近邻算法

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
训练
"""
# 包含(x,y)值的25个已知/训练数据的特征集
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
print(trainData)

# 用数字0和1分别标记红色或蓝色
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)
print(responses)
# 取红色族并绘图
red = trainData[responses.ravel() == 0]
print(red)
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

# 取蓝色族并绘图
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')


"""
验证
"""
newcomer = np.random.randint(0, 100, (10, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')  # 红色0
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')  # 蓝色1
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)
print("result: {}\n".format(results))
print("neighbours: {}\n".format(neighbours))
print("distance: {}\n".format(dist))
plt.show()
