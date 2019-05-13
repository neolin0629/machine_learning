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





## Reference

[How to Develop Convolutional Neural Networks for Multi-Step Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)

[How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)