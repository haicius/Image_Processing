# Author: 海子
# Create Time: 2021/8/18
# FileName: surf
# Descriptions:

import cv2
import numpy as np

img = cv2.imread('img/03.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t1 = cv2.getTickCount()
surf = cv2.xfeatures2d.SURF_create(500)   # 参数500是海森阈值，也可通过下面代码更改

# 检查以及修改海森矩阵阈值
print(surf.getHessianThreshold())
# 我们将其设置为50000。记住，它仅用于表示图片。
# 在实际情况下，最好将值设为300-500
surf.setHessianThreshold(400)
# 再次计算关键点并检查其数量

# 查看修改描述符维数，64和128维
print(surf.descriptorSize(), surf.getExtended())     # 默认的描述为64维
surf.setExtended(True)   # 修改128

# U-SURF
print(surf.getUpright())
surf.setUpright(True)
kp, des = surf.detectAndCompute(gray, None)
t2 = cv2.getTickCount()
time = (t2 - t1)/cv2.getTickFrequency()
print("SURF-->time=%s ms" % (time*1000))
print("kp", len(kp))

img = cv2.drawKeypoints(gray, kp, img,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                        )
cv2.imshow("surf", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
