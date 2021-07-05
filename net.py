import tensorflow
from tensorflow.keras import layers,Sequential
from tensorflow.keras import optimizers,losses
import matplotlib.pyplot as plt

#网络容器，创建模型
net=Sequential([
    layers.Conv2D(6,kernel_size=3,strides=1,padding="SAME"),#第一个卷积层，6个3*3卷积核
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2,strides=2),#高宽各减半的池化层
    layers.ReLU(),#激活函数
    layers.Conv2D(16,kernel_size=3,strides=1,padding="SAME"),#第二个卷积层，16个8*8卷积核
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2,strides=2),#高宽各减半的池化层
    layers.ReLU(),#激活函数
    layers.Flatten(),#打平以便进入全连接层
    layers.Dense(120,activation='relu'),#全连接层，120节点
    layers.Dense(84,activation='relu'),#全连接层，84节点
    layers.Dense(10)#10输出节点，分别是属于每个手势的概率
])
net.build(input_shape=(128,32,32,3))
net.summary()#查看网络参数信息

#采用Adam优化器，学习率0.01，采用交叉熵损失函数，包含Softmax
net.compile(optimizer=optimizers.Adam(lr=0.01),
           loss=losses.CategoricalCrossentropy(from_logits=True),
           metrics=['accuracy']
           )

#模型训练
history=net.fit(train_db,epochs=70)
history.history

#可视化训练结果
list1=[]
for i,j in history.history.items():
    list1.append(j)

plt.title('loss')
plt.grid(linestyle='-.')
plt.plot(range(1,71),list1[0], c='red')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(linestyle='-.')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(range(1,71),list1[1], c='red')

#模型测试
net.evaluate(test_db)

#保存网络参数
net.save_weights('weights.ckpt')
print('saved weighs!')