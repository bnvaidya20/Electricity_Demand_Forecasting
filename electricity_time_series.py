
"""
Electricity Demand Forecasting using Time Series
"""

# Import Libraries
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

from pylab import rcParams
rcParams['figure.figsize'] = (10, 6)

from statsmodels.tsa.seasonal import seasonal_decompose

from handler import Plotter, Stationarity, CorrFunction, Model_fitting, Prediction


for dirname, _, filenames in os.walk('./Electricity_Demand_Forecasting'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the dataset
        
file_path = "./data/Electric_Production.csv"

df = pd.read_csv(file_path, parse_dates=['DATE'])

print(df.head())
print(df.info())

# Data Transformation

df.columns=['Date', 'Production']

# Check for nulls
print(df.isna().sum())

df=df.dropna()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True) # set date as index

print(df.head())
print(df.info())
print(df.describe())

pt=Plotter()

# Visualizing the time series.
pt.plot_time_series(df)

# Plot the scatterplot:
pt.plot_scatterplot(df)

# Visualize the data in the series through a distribution.
pt.plot_distribution(df)


# Multiplicative Decomposition (Separating Trend and Seasonality from the time series).
result_mult = seasonal_decompose(df, model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df, model='additive', extrapolate_trend='freq')

# Plot Multiplicative & Additive
pt.plot_decompose(result_mult, "Multiplicative Decomposition")
pt.plot_decompose(result_add, "Additive Decomposition")


# Concat Components 
df_reconstruct = pd.concat([result_mult.seasonal, result_mult.trend, result_mult.resid, result_mult.observed], axis=1)
df_reconstruct.columns = ['season', 'trend', 'residue', 'actual_values']
print(df_reconstruct.head())

sta = Stationarity()

# Perform the Dickey-Fuller test (ADFT) for stationarising the time series.
sta.test_stationarity_adft(df)


# Compute the log of the series, and the rolling average of the series. 

df_log, moving_avg, std_dev = sta.get_stationary_series(df)

sta.plot_stationary_series(df)


# Take the difference of the series and the mean at every point in the series.

df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)
print(df_log_moving_avg_diff.head())


# Perform the Dickey-Fuller test (ADFT) to check whether the data is stationary or not.
sta.test_stationarity_adft(df_log_moving_avg_diff)


# Check the weighted average to understand the trend of the data in time series. 

weighted_average = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
print(weighted_average.head())

# Plot exponential moving average (EMA) - weighted average of the last n prices, where the weighting decreases exponentially with each previous price/period. 

pt.plot_mplex(df_log, weighted_average, 'Logged', 'Weighted')

# Subtracting df_log with weighted_average and perform the Dickey-Fuller test (ADFT)

logScale_weightedMean = df_log - weighted_average

sta.test_stationarity_adft(logScale_weightedMean)


# Perform differencing by subtracting the previous observation from the current observation.

df_log_diff = df_log - df_log.shift()

pt.plot_time_series(df_log_diff, title="Shifted timeseries")


df_log_diff.dropna(inplace=True)

# Perform the Dickey-Fuller test (ADFT)

sta.test_stationarity_adft(df_log_diff)

# Perform decomposition 

result = seasonal_decompose(df_log, model='additive', period = 12)

pt.plot_decompose(result, "Additive Decomposition")

trend = result.trend
trend.dropna(inplace=True)
seasonality = result.seasonal
seasonality.dropna(inplace=True)
residual = result.resid
residual.dropna(inplace=True)

# Perform the Dickey-Fuller test (ADFT) .

sta.test_stationarity(residual)


# Finding the best parameters for our model
# Compute and Plot Autocorrelation Function(ACF) and Partial Autocorrelation Function(PACF) .

cf= CorrFunction(df_log_diff)

cf.plot_acf_pacf()

# Fitting model
# Fit ARIMA model and plot for order=(2,0,2).

mfit=Model_fitting(df_log_diff)

result_AR = mfit.fitting_arima_model()
print(result_AR.summary())

mfit.plot_model()

# Fit ARIMA model and plot for order=(2,1,0).
# result_AR = mfit.fitting_arima_model(order=(2,1,2))
# print(result_AR.summary())
# mfit.plot_model()

# Fit ARIMA model and plot for order=(3,1,3).
# result_AR = mfit.fitting_arima_model(order=(3,1,3))
# print(result_AR.summary())
# mfit.plot_model()

# Predictions to forecast electricity production for the next 6 years

pred= Prediction()

pred_original= pred.predict_diff_ts(result_AR, df_log)
forecast = pred.get_forecast_with_diff_arima(result_AR, df_log, 72)

# Plot Prediction
pred.plot_prediction(df, pred_original, forecast)

pred.plot_forecast(df, forecast)


# Forecast using SARIMA

model_fit=mfit.fitting_sarima_model(df, order=(1, 1, 1), seasonal_order=(1,1,1,12))
print(model_fit.summary())


forecast_sm=pred.get_forecast_sarima(df, model_fit, 72)
print(forecast_sm.shape)


df_final=pred.get_concat_org_pred(df, forecast_sm)

print(df_final.shape)


forecast_sm['Date'] = pd.to_datetime(forecast_sm['Date'])
forecast_sm.set_index('Date', inplace=True)

pred.plot_forecast(df, forecast_sm)




