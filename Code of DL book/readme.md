# Deep Learning

## chapter 6

**损失函数(loss function)**
为了简化用到的数学，使用均方误差损失函数。
$$
J(θ)=\frac{1}{4}\underset{x\in X}{\sum}(f^*(x)-f(x;\Theta))^2
$$

>SSE(和方差)
>该参数计算的是拟合数据和原始对应点的误差的平方和
>
>![](http://ww1.sinaimg.cn/large/006WFczGgy1g2zl0y55yxj3082017q2q.jpg)
>MSE(均方方差)
>该统计参数是预测数据和原始数据对应点误差的平方和的均值，也就是SSE/n，和SSE没有太大区别
>
>![](http://ww1.sinaimg.cn/large/006WFczGgy1g2zl1fqa5qj30av01j3yc.jpg)

**激活函数（activation function）**：通常是用来扩展线性模型为非线性的，常见的有tanh, sigmiod, ReLU

>ReLU：线性整流函数（Rectified Linear Unit, ReLU）,又称修正线性单元
>常用整流函数有：f(x) = max(0, x)

## 基于梯度的学习
**交叉熵(cross-entropy)**
在信息论中，基于相同事件测度的两个概率分布**p**和**q**的交叉熵是指，当基于一个“非自然”（相对于“真实”分布**p**而言）的概率分布**q**进行编码时，在事件集合中唯一标识一个事件所需要的平均比特数（bit）。

基于概率分布**p**和**q**的交叉熵定义为：![](http://ww1.sinaimg.cn/large/006WFczGgy1g2zl37y9nnj309800t743.jpg)，其中H(p)是p的熵，D是q到p的KL散度（也被称为p相对于q的相对熵）。

> 均方误差和平均绝对误差在使用基于梯度的优化方法时成效不佳，即使在部分估计统计量的时候。

### 使用最大似然学习条件分布
[学习最大似然(视频)](https://www.bilibili.com/video/av15944258/?spm_id_from=333.788.b_7265636f5f6c697374.2)

大多数现代的神经网络使用最大似然来训练。这意味着代价函数就是负的对数似然，它与训练数据和模型分布间的交叉熵等价。
使用最大似然来估计参数的好处：
1. 明确一个模型p(y|x)则自动确定了一个代价函数logp(y|x)
2. 可以避免因为输出单元包含指数函数时，在它的变量取绝对值非常大的负值时会造成饱和的情况

### 输出单元