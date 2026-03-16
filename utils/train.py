import cv2
import numpy as np
import pandas as pd
import tflearn
from tflearn import max_pool_2d, conv_2d, input_data, fully_connected, regression
from torch.nn.functional import conv1d

train_data = np.load('hat_none.npy', allow_pickle=True)

# 拆分数据集
train = train_data[:-30]
test = train_data[-30:]

# 转成一维数据
x_train = np.array([line[0] for line in train]).reshape([-1, 50, 50, 1])
y_train = np.array([line[1] for line in train])
x_test = np.array([line[0] for line in test]).reshape([-1, 50, 50, 1])
y_test = np.array([line[1] for line in test])

# 搭建网络
# 输入层
input_size = 50
conv_input = input_data([None, input_size, input_size, 1], name='input')

conv1 = conv_2d(conv_input, 16, 3, activation='relu')
conv1_p1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(conv1_p1, 32, 3, activation='relu')
conv2_p1 = max_pool_2d(conv2, 2)

conv3 = conv_2d(conv2_p1, 64, 3, activation='relu')
conv3_p1 = max_pool_2d(conv3, 2)

conv4 = conv_2d(conv3_p1, 128, 3, activation='relu')
conv4_p1 = max_pool_2d(conv4, 2)

fully_layer1 = fully_connected(conv4_p1, 1024, activation='relu')
fully_layer2 = fully_connected(fully_layer1, 2, activation='softmax')

model_net = regression(
    fully_layer2,
    optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=0.001,
    name='model_net')
model = tflearn.DNN(model_net, tensorboard_dir='log')

model.fit({'input': x_train}, {'model_net': y_train},
          n_epoch=500, batch_size=100,
          validation_set=({'input': x_test}, {'model_net': y_test}),
          snapshot_step=5, show_metric=True, run_id='model_net')

model.save('../model/anquanmao.model')

# img_path = 'dog.jpg'
#
# img = cv2.imread(img_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img_gray, (50, 50))
# res = model.predict(img.reshape(1, 50, 50, -1))
# print(res)

