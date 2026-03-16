import tflearn
from tflearn import regression, input_data, fully_connected, max_pool_2d, conv_2d


conv_input = input_data([None,50,50,1],name='input')
cov1 = conv_2d(conv_input,32,3,activation='relu')
cov1 = max_pool_2d(cov1,2)
print(cov1.shape)
# # # 隐藏层2
cov2 = conv_2d(cov1,64,3,activation='relu')
cov2 = max_pool_2d(cov2,2)
# #
# # # 隐藏层3
cov3 = conv_2d(cov2,128,3,activation='relu')
cov3 = max_pool_2d(cov3,2)
# #
cov4 = conv_2d(cov3,256,3,activation='relu')
cov4 = max_pool_2d(cov4,2)
print(cov4.shape)
# # # #
cov5 = conv_2d(cov4,512,3,activation='relu')
cov5 = max_pool_2d(cov5,2)
full_layer = fully_connected(cov5,1024,activation='relu')
# #全连接层2
full_layer2 = fully_connected(full_layer,2,activation='softmax')
#
model_net = regression(full_layer2,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.00001,name='model_net')
# #创建
model =tflearn.DNN(model_net,tensorboard_dir='log')
model.load("./model/anquanmao.model")
import cv2
def model_pre(frame):
    try:
        # print(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #img  不是图片是矩阵数据[]
        frame = cv2.resize(frame,(50,50))
        pre=model.predict(frame.reshape(1,50,50,1))
        # print(pre)
        if pre[0][0]< pre[0][1]:  #  0.2 < 0.8
            print("hat")
            return "hat"
        else:
            print("none")
            return "none"
    except Exception as e:
        print(e)

def model_test(img):
    frame = cv2.imread(img)
    try:
        print(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #img  不是图片是矩阵数据[]
        frame = cv2.resize(frame,(50,50))
        pre=model.predict(frame.reshape(1,50,50,1))
        print(pre)
        if pre[0][0]<pre[0][1]:  #  0.2 < 0.8
            print("hat")
            return "hat"
        else:
            print("none")
            return "none"
    except Exception as e:
        print(e)

# model_test("hat.jpg")