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

#### Structure of model

| Layer   | Output | Shape | Params |
| ------- | ------ | ----- | ------ |
| input_1 | input  |       |        |



## Reference

[How to Develop Convolutional Neural Networks for Multi-Step Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)

[How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

