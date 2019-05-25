# Household power consumption

[TOC]

## Target

* How to develop a CNN for multi-step time series forecasting model for univariate data.
* How to develop a multichannel multi-step time series forecasting model for multivariate data.
* How to develop a multi-headed multi-step time series forecasting model for multivariate data.

## Data description

- **global_active_power**: The total active power consumed by the household (kilowatts).
- **global_reactive_power**: The total reactive power consumed by the household (kilowatts).
- **voltage**: Average voltage (volts).
- **global_intensity**: Average current intensity (amps).
- **sub_metering_1**: Active energy for kitchen (watt-hours of active energy).
- **sub_metering_2**: Active energy for laundry (watt-hours of active energy).
- **sub_metering_3**: Active energy for climate control systems (watt-hours of active energy).
- **A fourth sub-metering**: (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3. 

## Problem Framing

Given recent power consumption, what is the expected power consumption for the week ahead?

We can calculate the sum of all observations for each day and create a new dataset of daily power consumption data for each of the eight variables.

## Evaluation Metric

It is common with multi-step forecasting problems to evaluate each forecasted time step separately. This is helpful for a few reasons:

- To comment on the skill at a specific lead time (e.g. +1 day vs +3 days).
- To contrast models based on their skills at different lead times (e.g. models good at +1 day vs models good at days +5).

The performance metric for this problem will be the RMSE for each lead time from day 1 to day 7.

> Root Mean Squared Error (RMSE, 均方根误差) and Mean Absolute Error (MAE, 平均绝对误差) fit this bill, although RMSE is more commonly used 

## CNNs for Multi-Step Forecasting

### Benefit of using CNN

* It can be used in either a **recursive** or **direct** forecast strategy, where the model makes one-step predictions and outputs are fed as inputs for subsequent predictions, and where one model is developed for each time step to be predicted. 
* It can support multiple 1D inputs in order to make a prediction. This can be achieved using two different model configurations.
  * **Multiple Input Channels**. This is where each input sequence is read as a separate channel, like the different channels of an image (e.g. red, green and blue).
  * **Multiple Input Heads**. This is where each input sequence is read by a different CNN sub-model and the internal representations are combined before being interpreted and used to make a prediction.

### Data preparation

Organizing the data into standard weeks.

1. train dataset: (159, 7 ,8)
2. test dataset: (46, 7, 8)
3. [samples, timesteps, features]

### Multi-step Time Series Forecasting With a Univariate CNN

> *Given some number of prior days of total daily power consumption, predict the next standard week of daily power consumption.*

#### Hyperparameter

epochs: 20, batch_size: 4, optimizer:Adam (implementation of stochastic gradient descent)

loss function: MSE, evaluation: RMSE

#### Structure of model

**conv1d**： filters: 16 , kernel_size: 3, strides: 1

**maxpooling1d**：pool_size: 2

**dense1**：units: 10

**dense2**：units: 7

| Layer        | Shape            | Params |
| ------------ | ---------------- | ------ |
| input_1      | (None, 7 ,1)     | 0      |
| conv1d       | (None, 5, 1, 16) | 49     |
| maxpooling1d | (None, 4, 1, 16) | 0      |
| flatten      | (None, 64)       | 0      |
| dense1       | (None, 10)       | 650    |
| dense2       | (None, 7)        | 77     |

**improve** : change [n_input] from 7 to 14

### Multi-step Time Series Forecasting With a Multichannel CNN

#### Hyperparameter

epochs: 70, batch_size: 16, optimizer:Adam (implementation of stochastic gradient descent)

loss function: MSE, evaluation: RMSE

> * The increase in the amount of data requires a larger and more sophisticated model that is trained for longer.
>
> * batch_size: 在计算梯度下降的时候，使用多少个样本。
>   * 较大的batch_size在决定下降的方向时越准确，但是消耗内存，并且由于不同batch的采样差异性，可能导致梯度修正值互相抵消，无法修正（梯度消失）。(批梯度下降，Batch gradient descent)
>   * 较小的batch_size能够带来更好的泛化误差，计算速度快，但是收敛性能不好。（随机梯度下降，stochastic gradient descent）
>   * 当batch_size设置为2的次幂时能够充分利用矩阵运算

#### Structure of model

**conv1d_1**： filters: 32 , kernel_size: 3, strides: 1

**conv1d_2**： filters: 32 , kernel_size: 3, strides: 1

**maxpooling1d_1**：pool_size: 2

**conv1d_3**： filters: 16 , kernel_size: 3, strides: 1

**maxpooling1d_2**：pool_size: 2

**dense1**：units: 100

**dense2**：units: 7

| Layer          | Shape             | Params |
| -------------- | ----------------- | ------ |
| input_1        | (None, 14, 8)     | 0      |
| conv1d_1       | (None, 12, 8, 32) | 800    |
| conv1d_2       | (None, 10, 8, 32) | 800    |
| maxpooling1d_1 | (None, 9, 8, 32)  | 0      |
| conv1d_3       | (None, 7, 8, 16)  | 400    |
| maxpooling1d_2 | (None, 6, 8, 16)  | 0      |
| flatten        | (None, 768)       | 0      |
| dense1         | (None, 100)       | 76900  |
| dense2         | (None, 7)         | 707    |

### Multi-step Time Series Forecasting With a Multihead CNN

**Advantage:**

Can use different kernel sizes on the same data – much like a ResNet type design.

## LSTM Models for Multi-Step Time Series Forecasting 



## Reference

[How to Develop Convolutional Neural Networks for Multi-Step Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)

[How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

[How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

[How to Develop LSTM Models for Multi-Step Time Series Forecasting of Household Power Consumption](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/)

