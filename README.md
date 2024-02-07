# Electricity Demand Forecasting using Time Series Analysis


## Introduction

In recent years, electricity consumption has become a primary concern for countries that have adopted it as a primary energy source. Electricity demand forecasting is an analytical technique used to predict future electricity usage levels. It helps utilities and other businesses understand where and when peak demand will occur.
 
As the world continues to move towards digitalization, the demand for electricity increases constantly. This makes it essential for companies that operate in this space to be able to analyze their electricity usage and forecast its demand accordingly, for risk managing purposes.

Time series analysis can be used to better understand the characteristics of electricity demand and forecast usage patterns. 

In this project, time series analysis is used to uncover hidden relationships in electricity production data, predict future demand, and identify trends. 

Pandas, ARIMA, and SARIMAX in Python have been used to perform the time series analysis and predict the future.

## Solution

- Programming language: Python
- Libraries: 
  - Numpy, Pandas, Matplotlib
  - Statsmodels: Seasonal Decompose, Adfuller, ACF, pACF
  - Statsmodels: ARIMA, SARIMAX

## Exploratory Data Analysis

The raw data for this project contains time series data of a single variable, the electricity production. The data is aggregated monthly from January 1st, 1985 to January 1st, 2018.

## Time Series Analysis

1. Visualizing the time series.

2. Multiple approaches have been taken to perform the time series analysis:
- Seasonal Decompose 
  - Additive decompose
  - Multiplicative decompose

3. Stationarising the time series.
- ADF (Augmented Dickey-Fuller) Test: Dickey-Fuller test is one of the most popular statistical tests. 

4. Finding the best parameters for the model
- Deploy Autocorrelation Function(ACF) and Partial Autocorrelation Function(PACF)

5. Fitting model & Prediction
- ARIMA

6. Fitting model & Prediction
- SARIMAX

## References

1. [Time series analysis - predicting the consumption of electricity in the coming future](https://www.kaggle.com/datasets/kandij/electric-production)
2. [Time Series Analysis in Python](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)