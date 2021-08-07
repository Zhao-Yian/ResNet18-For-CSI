
This is a neural network written based on the ResNet18 architecture for training and predicting performance on pre-processed CSI channel data.

matlab是用于处理csi-tool收集的二进制csi信号的代码，可以生成相应的mat文件，方便predict.py进行后续处理.
main.py是训练代码，其中调用了tensorboard回调函数，可以保存网络参数并实时监控训练进度.
