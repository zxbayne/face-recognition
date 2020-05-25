# Face Recognition Demo
适合新手进行参考的人脸识别项目

![0]

[0]: https://img.shields.io/badge/language-chinese-green "language"

## 项目细节
### 1.数据采集
[capture.py](https://raw.githubusercontent.com/zxbayne/FaceRecognitionDemo/master/capture.py)

执行`capture.py`，通过`opencv-python`调用系统摄像头，使用`haarcascade`人脸特征提取器识别人脸。
将会建立两个文件夹`faceImages`与`faceImagesGray`。目录结构如下:
```
Project Folder
|---faceImages
    |---person1
        |---1.jpg
        |---2.jpg
        ...
        |---2000.jpg
    |---person2
    |---person3

|---faceImagesGray
    |---person1
        |---1.jpg
        |---2.jpg
        ...
        |---2000.jpg
    |---person2
    |---person3
```
`faceImages`目录仅仅保存拍摄的原图，不起任何作用。`faceImagesGray`则用于网络的训练。
### 2.数据预处理
[transform.py](https://raw.githubusercontent.com/zxbayne/FaceRecognitionDemo/master/transform.py)

执行`transform.py`，将`faceImagesGray`文件夹内的图片转换为`numpy.ndarrary`对象并持久化存储。存放于
`dataset`文件夹内。


生成的文件：
```
images.npy:人脸的图像，shape:[图片数目, 144, 144]
labels.npy:人脸图像对应的标签 shape:[图片数目]
relation.json: 标签与本人的对应关系
```
### 3.模型训练
[train.py](https://raw.githubusercontent.com/zxbayne/FaceRecognitionDemo/master/train.py)

执行`train.py`，搭建一个用于人脸检测的CNN神经网络模型，将训练集样本放入模型进行训练，
用测试集样本进行模型性能测试，并将训练好的模型进行保存。
生成`model`存放训练好的网络模型，`logs`存放日志文件。
在命令行内输入`tensorboard --logdir /path/to/your/logs`，然后打开浏览器，
输入localhost:6006，即可可视化损失函数值与精确度。

### 4.模型测试
[test.py](https://raw.githubusercontent.com/zxbayne/FaceRecognitionDemo/master/test.py)

执行`test.py`，将提取到的脸部图片送入网络模型，将模型的结果实时呈现。
