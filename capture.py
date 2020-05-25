"""
从视频或者摄像头中获取指定数量的人脸图像并
保存，用于训练模型
"""

import os
import shutil

import cv2

# 存储原始图像的目录
RAW_PATH = './faceImages'
# 存储人脸灰度图的目录
GRAYSCALE_PATH = './faceImagesGray'
# 视频源，一个具体的路径（/home/user1/1.mp4） 也可以是整数（0是获取笔记本的摄像头）
VIDEO_SOURCE = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# 初始化VideoCapture类与人脸特征提取器
cap = cv2.VideoCapture(VIDEO_SOURCE)
classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def capture(source, img_num, raw_path, grayscale_path):
    """
    :param source: 视频源
    :param img_num: 需要获取的人脸图像数目
    :param raw_path: 原始图片的保存路径
    :param grayscale_path: 人脸灰度图的保存路径
    :return: 无
    """
    # 记录已经截取的人脸图像数目
    img_count = 0

    # 开始截取
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # 转换为灰度图
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classifier.detectMultiScale(grayscale_image, 1.3, 5)
        # 如果len(faceRects > 0，则检测到人脸
        if len(faceRects) > 0:
            # 遍历视频流里的每张脸
            for face in faceRects:
                img_count += 1
                # 获取脸的坐标
                x, y, w, h = face
                # 绘制脸所在的位置
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0, 255), 2)
                # 实时显示已截取脸的图片的数目以及目标数目
                cv2.putText(frame, "{}/{}".format(img_count, img_num), (x - 40, y - 20), font, 1, (0, 0, 255), 2)

                # 保存图片
                raw_filename = "{}/{}.jpg".format(RAW_PATH, img_count)
                grayscale_filename = "{}/{}.jpg".format(GRAYSCALE_PATH, img_count)
                cv2.imwrite(raw_filename, frame)
                cv2.imwrite(grayscale_filename, grayscale_image[y: y + h, x: x + w])

        # 使用opencv可视化
        cv2.imshow("Capture", frame)
        c = cv2.waitKey(10)
        # 按q键退出
        if c & 0xFF == ord('q'):
            break
        if (img_count >= img_num):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input('输入截取者的姓名（拼音）:')
    img_num = int(input("需要截取的人脸图像数目 :"))

    # 建立图片的根目录
    try:
        os.mkdir(path=RAW_PATH)
        os.mkdir(path=GRAYSCALE_PATH)
    except FileExistsError:
        pass
    # 建立截取者的目录
    RAW_PATH = os.path.join(RAW_PATH, name)
    GRAYSCALE_PATH = os.path.join(GRAYSCALE_PATH, name)
    try:
        os.mkdir(RAW_PATH)
        os.mkdir(GRAYSCALE_PATH)
    except FileExistsError:
        shutil.rmtree(RAW_PATH)
        shutil.rmtree(GRAYSCALE_PATH)
        os.mkdir(RAW_PATH)
        os.mkdir(GRAYSCALE_PATH)

    capture(VIDEO_SOURCE, img_num, RAW_PATH, GRAYSCALE_PATH)
