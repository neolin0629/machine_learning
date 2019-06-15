# RNN
RNN（Recurrent Neural Network）是一类用于处理序列数据的神经网络。基础的神经网络只在层与层之间建立了权连接，RNN最大的不同之处就是在层之间的神经元之间也建立的权连接。
这是一个标准的RNN结构图，图中每个箭头代表做一次变换，也就是说箭头连接带有权值。左侧是循环图，右侧是展开图，左侧中h旁边的黑色方块表示每个时间步。 

![](pics\c10_rnn_unfolding.png)

> 图中O代表输出，y代表样本给出的确定值，L代表损失函数，“损失”也是随着序列的推进而不断积累的。 
> 从t=1到t=n的每个时间步，**更新方程**如下
> ![](pics\c10_rnn_formula.png)
> 由于RNN常用于分类，所以o的激活函数一般用softmax


**展开图(unfold)**引入了两个主要优点：

1. 无论序列的长度，学成的模型始终具有相同的输入大小，因为它指定的是从一种状态到另一种状态的转移，而不是在可变长度的历史状态上操作。

  ![](pics\c10_rnn_unfold_formula.png)

2. 可以在每个时间步使用相同参数的相同转移函数f，且权值共享，图中的W全是相同的，U和V也一样。 

## 经典的RNN结构
经典的RNN结构
![](pics\diags.jpeg)
> 1. image classification
> 2. image captioning takes an image and outputs a sentence of words
> 3. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment
> 4. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French
> 5. Synced sequence input and output(e.g. video classification where we wish to label each frame of the video). 

## Encoder-Decoder结构

Encoder-Decoder，也可以称之为Seq2Seq结构，该架构输入和输出的长度可以不同。具体过程如下：
1. 编码器(encoder)或读取器(reader)或输入(input)RNN处理输入序列，编码器输出上下文C(通常是最终隐藏状态h(nx)的简单函数)
2. 解码器(decoder)或写入器(writer)或输出(output)RNN则以固定长度的向量(如图10.9)为条件产生输出序列Y=(y(1), . . . , y(ny))

如果C是一个向量，则decoder是一个向量到序列的RNN。这时，向量的输入有两种方式，这两种方式也可以结合
1. 作为RNN的初始状态
![](pics\encoder_decoder.jpg)
2. 连接到每个时间步的隐藏单元
![](pics\encoder_decoder_connect_2_t.jpg)


## RNN的训练方法（BPTT）
**BPTT(back-propagation through time)**算法是常用的训练RNN的方法，其实本质还是BP算法，只不过RNN处理时间序列数据，所以要基于时间反向传播，故叫**通过时间反向传播**。BPTT的中心思想和BP算法相同，沿着需要优化的参数的负梯度方向不断寻找更优的点直至收敛。综上所述，BPTT算法本质还是BP算法，BP算法本质还是梯度下降法，那么求各个参数的梯度便成了此算法的核心。

根据上面的更新方程，每个节点的参数有U,V,W,b和c，以及以t为索引的节点序列x(t),h(t),o(t)和L(t)。每个参数的梯度计算公式如下：(*)

<img src="pics\BPTT_gradient_fomula.png" width="70%" height="70%"/>

### 梯度消失和梯度爆炸
参考：[RNN梯度消失和爆炸的原因](https://zhuanlan.zhihu.com/p/28687529)

#### 原因
梯度消失(gradient vanishing)会导致网络权重基本不更新(梯度消失的那一层变为单纯的映射层)，从而造成训练困难。梯度爆炸(exploding gradient)会导致网络权重的大幅更新，使网络变得不稳定，极端情况，权重的值非常大，以至于溢出，导致NaN值。

在任意时刻，对参数W，U，V求偏导，因为L是随着时间累加的，所以整体的损失等于每一时刻损失值的累加。其中V只关注当前：

![](pics\derivative_v.png)

W, U需要追溯之前的历史数据：

![](pics\derivative_w_u.png)

激活函数是嵌套在里面的，即![](pics\hj_fomula.png)
所以中间累乘的那部分可以转换为：![](pics\derivative_tanh.png)
或是：![](pics\derivative_sigmoid.png)
我们会发现累乘会导致激活函数导数的累乘，进而会导致梯度消失和梯度爆炸现象的发生。

这是sigmoid函数的函数图和导数图。

<img src="pics\derivative_sigmoid_pic.png" width="50%" height="50%"/>

这是tanh函数的函数图和导数图。

<img src="pics\derivative_tanh_pic.png" width="50%" height="50%"/>

sigmoid函数的导数范围是(0,0.25]，tanh函数的导数范围是(0,1]，他们的导数最大都不大于1。在长期依赖的情况下，如果W也是一个大于0小于1的值，则当t很大时，累乘会趋近于0，这就是梯度消失现象。同理，当W很大时，累乘就会趋近于无穷，这就是梯度爆炸现象。

#### 应对
一般来说，梯度爆炸相对于梯度消失，不是个严重的问题。
1. 梯度爆炸是在一个更窄范围内发生的问题；梯度消失更为普遍
2. 梯度爆炸相对于梯度消失更容易观察到，比如模型不稳定，更新过程中损失出现显著变化或者出现溢出（NaN）；梯度消失不容易观察且更难以处理

解决梯度消失的办法一般有：
1. 选取更好的激活函数
> ReLU的导数值为0或者1，可以避免梯度消失，但是恒为1时，有可能产生梯度爆炸
>
> <img src="pics\derivative_relu_pic.png" width="50%" height="50%"/>
2. 使用LSTM或者GRU等结构

解决梯度爆炸的办法一般有：
1. 使用梯度截断。
> 在非常深且批尺寸较大的多层感知机网络和输入序列较长的LSTM中，仍然有可能出现梯度爆炸，需要训练过程中检查和限制梯度的大小。具体来说，检查误差梯度的值是否超过阈值，如果超过，则截断梯度，将梯度设置为阈值。
2. 使用权重正则化
> 检查网络权重的大小，并惩罚产生较大权重值的损失函数。该过程被称为权重正则化，通常使用的是L1惩罚项（权重绝对值）或L2惩罚项（权重平方）。

#### 批标准化(batch normalization)
机器学习领域有个很重要的假设：IID(independent and identically distributed)独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。那BN的作用是什么呢？BN就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

BN是基于mini-batch SGD(小批量随机梯度下降)的优化。
##### 1. Internal Covariate Shift问题
covariate shift的概念：如果ML系统实例集合<X,Y>中的输入值X的分布老是变，这不符合IID假设，网络模型很难稳定的学规律。在训练过程中，因为各层参数不停在变化，所以每个隐层都会面临covariate shift的问题，这就是所谓的"Internal Covariate Shift"。

##### 2. BN的本质
**对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。**

举几个例子：
sigmoid作为激活函数

<img src="pics\BN_sigmoid.png" width="50%" height="50%"/>

<img src="pics\BN_sigmoid_2.png" width="50%" height="50%"/>

relu作为激活函数

<img src="pics\BN_relu.jpg" width="50%" height="50%"/>

##### 3. BN的做法
![](pics\BN_operation.png)

1. 第三步的ε是一个常数（趋于0），在不影响归一化的同时，避免除数出现0的情况，保证计算方差时的稳定性
2. 一般的归一化在第三步就结束了，最后一步是因为归一化后的xi基本会被限制在正态分布下，使得网络的表达能力下降，引入可学习的γ,β进行scale和shift操作可以让模型在训练的过程中自己选择最适合的分布。

##### 4. BN在预测时的使用
预测时，我们一般只输入一个测试样本(即没有了mini-batch)，观察结果，这时候的均值u(/mju:/)、标准差σ(/'sɪɡmə/)怎么算？这里我们用每个训练batch的u、σ来求整个训练样本的u、σ，作为预测时进行BN的均值和方差。

<img src="pics\BN_total_u_sigma.png" width="25%" height="25%"/>

上面简单理解就是：对于均值来说直接计算所有batch u值的平均值；然后对于标准偏差采用每个batch σB的无偏估计。

最后测试阶段，BN的使用公式如下：

<img src="pics\BN_interence_fomula.png" width="50%" height="50%"/>

这个公式其实和训练时的公式四等价，通过简单的合并计算推导就可以得出。

##### 5. BN的好处
1. 极大提升了训练速度，收敛过程大大加快
2. 增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果
3. 调参过程简单，对于初始化要求没那么高，而且可以使用大的学习率(learning rate)等

## RNN的不同逻辑结构
RNN中一些重要的设计模式包括以下几种：
1. 每个时间步都有输出，并且隐藏单元之间有循环连接的循环网络，如经典的RNN结构
2. 每个时间步都产生一个输出，只有当前时刻的输出到下个时刻的隐藏单元之间有循环连接的循环网络，如下图1
3. 隐藏单元之间存在循环连接，但读取整个序列后产生单个输出的循环网络，如下图2

图1：![](pics\c10_rnn_o2h_recurrent.png)

特点**：循环链接是从输出o到隐藏层h。没有h到h的模型强大(只能表示更小的函数集合)。o作为输出，除非维度很高，否则会损失一部分h的信息。没有直接的循环链接，而只是间接的将h信息传递到下一层。但其易于训练，可以并行化训练(使用标注y^t替换o^t而作为传递到后面的信息，这样就不再需要先计算前一时间步的隐藏状态，再计算后一步的隐藏状态，因此所有计算都能并行化)。

图2：

![](pics\c10_rnn_n_1.png)

**特点**：先读取整个序列，然后再产生单个输出，循环连接存在于隐藏单元之间。这种架构常用于阅读理解等序列模型。这种架构只在最后一个隐藏单元输出观察值并给出预测，它可以概括序列并产生用于进一步运算的向量，例如在编码器解码器架构中，它可用于编码整个序列并抽取上下文向量。

### 导师驱动过程（teacher forcing）
Teacher Forcing是一种用来训练循环神经网络模型的方法，这种方法以上一时刻的输出作为下一时刻的输入。该方法最初是作为BPTT的替代技术的。这种类型的模型在语言模型中很常见，即使用正确的单词作为输入的一部分去预测下一个单词。

#### 原理
训练模型时，导师驱动过程不再使用最大似然准则，而在时刻t + 1接收真实值y(t)作为输入。条件最大似然准则是：

![](pics\c10_rnn_probability_fomula.png)

只考虑两个时间步的序列。取对数后：

![](pics\c10_tforce_likelihood.png)

在这个例子中，同时给定迄今为止的x序列和来自训练集的前一y值，我们可以看到在时刻t = 2时，模型被训练为最大化y(2)的条件概率。因此最大似然在训练时指定正确反馈，而不是将自己的输出反馈到模型。

#### 应用
![](pics\c10_rnn_teacher.png)

如上图所示：训练时，我们将训练集中正确的输出y(t)反馈到h(t+1)。当模型部署后，真正的输出通常是未知的。在这种情况下，我们用模型的输出o(t)近似正确的输出y(t)，并反馈回模型。

#### 改进
如果之后神经网络在开环(open-loop)模式下使用，即测试集中出现了训练集中不存在的数据时，模型的效果会不好。改进的方法有如下几种：
1. 搜索候选输出序列。在预测的是离散值时，通常可以使用集束搜索(beam search)。比如在预测单词这种离散值的输出时，一种常用方法是对词表中每一个单词的预测概率执行搜索，生成多个候选的输出序列。
2. 当模型预测的是实值(real-valued)而不是离散值(discrete value)时，使用课程学习策略(Curriculum Learning)，即使用一个概率p去选择使用ground truth的输出y(t)还是前一个时间步骤模型生成的输出o(t)作为当前时间步骤的输入。这个概率p会随着时间的推移而改变，这就是所谓的计划抽样(scheduled sampling)，训练过程会从force learning开始，逐步使用更多生成值作为输入。

## Reference

1. [RNN](https://blog.csdn.net/zhaojc1995/article/details/80572098)
2. [激活函数导数,取值图片来源](https://nn.readthedocs.io/en/rtd/transfer/)
3. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
4. [深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html)