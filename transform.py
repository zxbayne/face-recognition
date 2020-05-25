"""
将获取到的人脸图片转换为numpy.ndarray对象并持久化储存
"""
import json
import os

import cv2
import numpy as np

# TODO: 请手动修改images_num的值为总图片数目
images_num = 10000
# 人脸图像存储位置
root_path = 'faceImagesGray/'
height, width = 144, 144
# 初始化
images = np.zeros([images_num, height, width])
labels = np.zeros([images_num])

# 用于索引images、labels
index = 0
"""
count 用于编码labels
比如这样的目录：
    -faceImagesGray(根目录）
        |---Amy (子目录)
        |---Bob
        |---Cindy
        ......
        |---Zero
    则会被映射为如下的关系
    {
        Amy  : 0,
        Bob  : 1,
        Cindy: 2,
        ....
        Zero : 25
    }
"""
count = 0
relation = {}

# 转换为numpy.ndarray对象
for person in os.listdir(root_path):
    relation[person] = count
    subfolder = os.path.join(root_path, person)
    for file in os.listdir(subfolder):
        img_path = os.path.join(subfolder, file)
        # grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images[index, :] = cv2.resize(img, (height, width))
        labels[index] = count
        index += 1
    count += 1
print(relation)
print(images.shape)
print(labels.shape)

# 保存
out_path = 'dataset/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

np.save(out_path + "images.npy", images)
np.save(out_path + "labels.npy", labels)
out = json.dumps(relation)
with open(out_path + "relation.json", 'w') as file:
    file.write(out)
