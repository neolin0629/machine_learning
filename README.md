# machine_learning
learning project for machine learning

## DL Chapter 6

1. XOR（深度前馈网络）
2. Tensorflow中tensor和computation graph的讨论
3. xor的总结
4. playground算法和数据集在python上的实现
   * 算法必须做到nn层数的设定和每层节点的设定，激活函数以及损失函数的选取
5. 在playground中将计算从网页上放到计算服务器中，即分割展示和运算

## DL Chapter 7

1. 令目标函数为抛物线函数：f(x,y)=(x-x0)^2+(y-y0^2), 用matplotlib在tensorflow里将以L1和L2为正则函数的目标函数的等高线图画出来
2. 参照以上项目，将优化过程画出来

## CNN, RNN(LSTM)

> 初步的想法是用cnn抽取时间序列中的特征（features），用rnn，以及基于rnn的模型（lstm, resnet, attention）对这些特征建立具有长记忆的模型。

1. stock2vec的文献阅读和数据准备
2. 实现 .\reference\papers\gcnn.pdf
3. household power consumption
   * 关于cnn与time series：
     * 参考文献：https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
     * 基于这篇文章，解决两个课题：单（univariate）时间序列和多（multichannel 和multihead）个时间序列。
   * 关于rnn和lstm：
     * 参考文献：https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
     * 课题：a) 基础rnn和lstm； b) cnn-lstm: 用cnn抽取特征，用lstm建立模型。
