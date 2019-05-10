# XOR 总结

1. 使用非线性模型来描述这个函数

   1. 做仿射变换得到h
   2. 在h上使用激活函数（relu, sigmoid）达到非线性的目的

2. XOR问题的解在损失函数[loss function]的全局最小点，可以使用梯度下降的优化办法

   1. 梯度下降算法的收敛点取决于参数的初始值

   2. 根据XOR的决策边界，可以确定1层hidden layer，2个神经元是合适的结构

      > 参考阅读：[在神经网络中应使用多少隐藏层/神经元](<https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/82393197>)

