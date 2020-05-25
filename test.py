import json

import cv2
import numpy as np
import tensorflow as tf
import time

# relation.json的位置
path = 'dataset/relation.json'
VIDEO_SOURCE = 0
# 初始化
cap = cv2.VideoCapture(VIDEO_SOURCE)
classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

with open(path, 'r') as file:
    relation = file.readlines()[0]
relation = json.loads(relation)
relation = {value: key for key, value in relation.items()}

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/face.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))

def predict(img):
    try:
        pic = cv2.resize(img, (144, 144))
        pic = pic.reshape([1, 144, 144, 1])
    except Exception:
        print("{} :error on resize img".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return
    graph = tf.get_default_graph()
    x_new = graph.get_tensor_by_name("x:0")
    y_new = graph.get_tensor_by_name("out:0")

    pre = sess.run(y_new, feed_dict={
        x_new: pic
    })
    return pre

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = classifier.detectMultiScale(grey_image, 1.3, 5)
    # len(faceRects > 0 意味着脸部检测到了
    if len(faceRects) > 0:
        for face in faceRects:
            x, y, w, h = face
            pre = predict(grey_image[y - 10: y + h + 10, x - 10: x + w + 10])
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            cv2.putText(frame, relation[np.argmax(pre)], (x - 40, y - 20), font, 1, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("Capture", frame)
    c = cv2.waitKey(10)
    # press q for exit
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sess.close()
