# Author: 海子
# Create Time: 2021/8/24
# FileName: OCR_KNN_2
# Descriptions: 字母识别knn
import cv2 as cv
import numpy as np

# 加载数据，转换器将字母转换为数字
data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch) - ord('A')})

# 将数据分为两个，每个10000个以进行训练和测试
train, test = np.vsplit(data, 2)

# 将火车数据和测试数据拆分为特征和响应
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# 初始化kNN, 分类, 测量准确性
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=1)
correct = np.count_nonzero(result == labels)
accuracy = correct * 100.0 / 10000
print(accuracy)
