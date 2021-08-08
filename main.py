# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import keras
import pathlib
from tensorflow.keras import losses


loaddata = np.load("./train/volAdatashaochu.npy",allow_pickle=True)
loadlabel = np.load("./train/volAlabelshaochu.npy",allow_pickle=True)
predict_data = np.load("./testdata/not_predict.npy",allow_pickle=True)
predict_label = np.load("./testdata/volAlabelshaochu.npy",allow_pickle=True)
s_x,s_y = loaddata.shape
p_x,p_y = predict_data.shape
zeros = np.ones((p_x,1))
if s_y > p_y:
    for i in range(s_y-p_y):
        predict_data=np.hstack((predict_data,zeros))
#predict_data = np.load("d:\\train\\volAdatashaochu.npy",allow_pickle=True)
#print (loaddata.shape)
#print(predict_data.shape)
loadlabel=loadlabel.astype('float64')     #转换为张量形式
loaddata=loaddata.astype('float64')
predict_data = predict_data.astype('float64')
loaddata=np.expand_dims(loaddata,axis=2)     #增加空白维度 便于传入卷积层
loaddata=np.expand_dims(loaddata,axis=3)
predict_data=np.expand_dims(predict_data,axis=2)
predict_data=np.expand_dims(predict_data,axis=3)#其实conv2d改成1d就可以了。。。

class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(),
                                        use_bias=False)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y




if __name__ == '__main__':
    model = ResNet18([2, 4, 4, 2])
    index = [i for i in range(len(loaddata))]
    np.random.shuffle(index)        #打散验证集与标签 防止产生记忆
    loaddata = loaddata[index]
    loadlabel = loadlabel[index]
    sgd = keras.optimizers.SGD(lr=0.02, decay=1e-7, momentum=0.9, nesterov=True)
    Nadam=tf.optimizers.Nadam(learning_rate=0.005,beta_1=0.9,beta_2=0.999,epsilon=1e-7,name='Nadam')
    model.compile(optimizer='Nadam',
                  loss=losses.CategoricalCrossentropy(),
                  metrics=['CategoricalAccuracy'])  #因为是独热码 采用 mse计算loss 和CategoricalAccuracy的评判标准


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./dir_wonder",
                                                          histogram_freq=1,
                                                          write_graph=False,
                                                          write_images=False,
                                                          write_grads=False)
    checkpoint_save_path = "./checkpoint/wonder.ckpt"
    #if os.path.exists(checkpoint_save_path + '.index'):
        #print('-------------load the model-----------------')
        #model.load_weights(checkpoint_save_path)
    #model.load_weights(r'C:\Users\zhaoy\Desktop\checkpoint\people.ckpt')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    history = model.fit(loaddata, loadlabel, batch_size=32, epochs=50000, validation_split=0.1, validation_freq=1,
                        callbacks=[cp_callback,tensorboard_callback])
    #model.summary()

    loadlabel1 = model.predict(loaddata)
    px, py = loadlabel1.shape
    index2 = np.argmax(loadlabel1,axis=1)
    d= np.argmax(np.bincount(index2))
    count2 = 0
    for i in range(len(index2)):
        if index2[i] == loadlabel[i][1]:
            count2 += 1
    print(count2)
    arr2 = count2/len(index2)
    print("共{:d}组数据,匹配正确{:d}组,训练集准确率：{:.2f}".format(len(index2),count2,arr2))


    predictlabel = model.predict(predict_data)
    px, py = predictlabel.shape
    index1 = np.argmax(predictlabel,axis=1)
    d= np.argmax(np.bincount(index1))
    count1 = 0
    for i in range(len(index1)):
        if index1[i] == predict_label[i][1]:
            count1 += 1
    print(count1)
    arr1 = count1/len(index1)
    print("共{:d}组数据,匹配正确{:d}组,测试集准确率：{:.2f}".format(len(index1),count1,arr1))
