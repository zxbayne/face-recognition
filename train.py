"""
获取数据集并训练出网络
"""

import json
import os

import numpy as np
import tensorflow as tf

# 数据集位置
path = 'dataset/'
# 学习率
learning_rate = 0.01
epoches = 20
img_height, img_width = 144, 144
batch_size = 50
# 总的数据集中，取80%作为训练集，20%为测试集
ratio = 0.8

tensorboard_log = './logs/'


# 打乱数据集
def shuffle(x, y):
    indice = np.random.permutation(x.shape[0])
    x = x[indice]
    y = y[indice]
    return x, y


# 按批次取数据
def batch_generator(x, y, batch_size):
    assert x.shape[0] == y.shape[0], (
        "shape not match"
    )
    x, y = shuffle(x, y)

    index = 0
    while True:
        if index + batch_size > x.shape[0]:
            index = 0
        start = index
        index += batch_size
        yield x[start: start + batch_size], y[start: start + batch_size]


# 加载数据集
def load_dataset(path):
    images = np.load(path + 'images.npy')
    labels = np.load(path + 'labels.npy')
    with open(path + 'relation.json', 'r') as file:
        relation = file.readlines()[0]
    relation = json.loads(relation)
    relation = {value: key for key, value in relation.items()}
    return images, labels, relation


def CNN(input):
    with tf.variable_scope("conv_layer1") as conv_layer1:
        conv1 = tf.layers.conv2d(
            inputs=input,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.variable_scope("conv_layer2") as conv_layer2:
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.variable_scope("conv_layer3") as conv_layer3:
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    with tf.variable_scope("conv_layer4") as conv_layer4:
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

        flatten = tf.reshape(pool4, [-1, 9 * 9 * 128])

    with tf.variable_scope("dense_layer1") as dense_layer1:
        dense1 = tf.layers.dense(
            inputs=flatten,
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.003)
        )
        dropout1 = tf.layers.dropout(
            inputs=dense1,
            rate=0.3,
            training=is_training
        )

    with tf.variable_scope("dense_layer2") as dense_layer2:
        dense2 = tf.layers.dense(
            inputs=dropout1,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.003)
        )
        dropout2 = tf.layers.dropout(
            inputs=dense2,
            rate=0.5,
            training=is_training
        )

    with tf.variable_scope("dense_layer3") as dense_layer3:
        logits = tf.layers.dense(
            inputs=dropout2,
            units=n_classes,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.003)
        )
    return logits


if __name__ == '__main__':
    if not os.path.exists('model'):
        os.mkdir('model')
    images, labels, relation = load_dataset(path)
    images, labels = shuffle(images, labels)
    n_images = len(images)
    # 划分训练集与测试集
    x_train = images[:int(n_images * ratio)].reshape([-1, img_height, img_width, 1])
    x_test = images[int(n_images * ratio):].reshape([-1, img_height, img_width, 1])
    y_train = labels[:int(n_images * ratio)].astype("int32")
    y_test = labels[int(n_images * ratio):].astype("int32")
    n_batch = len(y_train) // batch_size
    n_batch_test = len(y_test) // batch_size
    # 类别数目
    n_classes = len(relation)

    # 对label进行one-hot编码
    y_train = np.eye(n_classes)[y_train]
    y_test = np.eye(n_classes)[y_test]

    x = tf.placeholder(tf.float32, [None, img_height, img_width, 1], name='x')
    y = tf.placeholder(tf.int32, shape=[None, n_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')

    model = CNN(x)
    model = tf.identity(model, 'out')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)), tf.float32))
    saver = tf.train.Saver()

    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    # 开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(tensorboard_log, graph=tf.get_default_graph())

        for current_epoch in range(epoches):
            train_set = batch_generator(x_train, y_train, batch_size)
            test_set = batch_generator(x_test, y_test, batch_size)
            for i in range(n_batch):
                batch_x, batch_y = next(train_set)

                _, loss, summay = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={
                                               x: batch_x,
                                               y: batch_y,
                                               is_training: True
                                           })
                summary_writer.add_summary(summay, current_epoch * n_batch + i)
                # 打印损失函数值
                if (i + 1) % 10 == 0:
                    print("epoch:{} batch:{}/{}:  loss:{}".format(
                        current_epoch + 1,
                        (i + 1) * batch_size,
                        int(n_images * ratio),
                        loss
                    ))
            # 使用测试集评估模型
            # 测试集数据量过大时，按批次读入进行评估可以节省内存
            acc_test = 0
            loss_test = 0
            for i in range(n_batch_test):
                batch_x_test, batch_y_test = next(test_set)
                # acc = accuracy.eval({x: batch_x_test, y: batch_y_test})
                acc,loss = sess.run([accuracy, cross_entropy], feed_dict={
                    x: batch_x_test,
                    y: batch_y_test,
                    is_training: False
                })
                loss_test += loss
                acc_test += acc
            print("===== epoch:{} loss:{} acc:{} =====".format(
                current_epoch + 1,
                loss_test / n_batch_test,
                acc_test / n_batch_test
            ))

        saver.save(sess, 'model/face')
