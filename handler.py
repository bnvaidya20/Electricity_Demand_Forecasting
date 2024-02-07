import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Plotter():
    def __init__(self):
        pass

    def plot_time_series(self, df, title="Electricity Production"):
        plt.xlabel("Date")
        plt.ylabel("Production")
        plt.title(title)
        plt.plot(df)
        plt.show(block=True)

    def plot_scatterplot(self, df):
        df.plot(style='k.')
        plt.show(block=True)

    def plot_distribution(self, df):
        df.plot(kind='kde')
        plt.show(block=True)

    def plot_decompose(self, result, title):
        plt.rcParams["figure.figsize"] = (12,5)
        result.plot().suptitle(title, fontsize=20)
        plt.show(block=True)

    @staticmethod
    def plot_mplex(df1, df2, label1, label2):
        plt.plot(df1, label=label1)
        plt.plot(df2, color='red', label=label2)
        plt.xlabel("Date")
        plt.ylabel("Production")
        plt.legend(loc='best')
        plt.show(block=True)


class Stationarity():
    def __init__(self):
        pass

    def test_stationarity_adft(self, df):
        # Determining rolling statistics
        rolmean = df.rolling(12).mean()
        rolstd = df.rolling(12).std()

        # Plot rolling statistics:
        plt.plot(df, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show()

        # Perform Dickey Fuller test  
        print("Results of Dickey fuller test:")
        adft = adfuller(df['Production'],autolag='AIC')
        # output for adft will give us without defining what the values are.
        # hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        print(output)

    def test_stationarity(self, df):
        #Determing rolling statistics
        rolmean = df.rolling(12).mean()
        rolstd = df.rolling(12).std()

        #Plot rolling statistics:
        plt.plot(df, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=True)
    
    def get_stationary_series(self, df):
        df_log = np.log(df)
        moving_avg = df_log.rolling(12).mean()
        std_dev = df_log.rolling(12).std()

        return df_log, moving_avg, std_dev

    def plot_stationary_series(self, df):  
        df_log, moving_avg, std_dev=self.get_stationary_series(df)
        plt.plot(df_log)
        plt.plot(moving_avg, color="red")
        plt.plot(std_dev, color ="black")
        plt.show()


class CorrFunction():
    def __init__(self, df):
        self.df = df

    def compute_acf_pacf(self):
        acfj = acf(self.df, nlags=15)
        pacfj = pacf(self.df, nlags=15,method='ols') 

        return acfj, pacfj

    # Plotting
    def plot_acf_pacf(self):
        acf, pacf = self.compute_acf_pacf()

        plt.subplot(121)# plot ACF
        plt.plot(acf) 
        plt.axhline(y=0,linestyle='-', color='blue')
        plt.axhline(y=-1.96/np.sqrt(len(self.df)), linestyle='--', color='black')
        plt.axhline(y=1.96/np.sqrt(len(self.df)), linestyle='--', color='black')
        plt.title('Auto corellation function')
        plt.tight_layout()
        plt.subplot(122)# plot PACF
        plt.plot(pacf) 
        plt.axhline(y=0,linestyle='-',color='blue')
        plt.axhline(y=-1.96/np.sqrt(len(self.df)), linestyle='--', color='black')
        plt.axhline(y=1.96/np.sqrt(len(self.df)), linestyle='--', color='black')
        plt.title('Partially autocorrelation function')
        plt.tight_layout()
        plt.show()

class Model_fitting():

    def __init__(self, df):
        self.df = df

    def fitting_arima_model(self, order=(2,0,2)):
        model = ARIMA(self.df, order=order)
        result_AR = model.fit()
        return result_AR
    
    @staticmethod
    def fitting_sarima_model(df, order, seasonal_order):
        model = SARIMAX(df, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        return model_fit

    def plot_model(self): 
        result_AR = self.fitting_arima_model()
        RSS = sum((result_AR.fittedvalues-self.df["Production"])**2)
        plt.plot(self.df)
        plt.plot(result_AR.fittedvalues, color='red')
        plt.title(f"Sum of squares of residuals  RSS: {RSS}")
        plt.show()


class Prediction():
    def __init__(self):
        pass

    def predict_diff_ts(self, fitted_model, df_log):
        pred_diff = fitted_model.fittedvalues # predict on the differenced data

        # Revert differencing
        pred_cumsum = pred_diff.cumsum()  # cumulative sum to revert differencing

        # Get the index of the first non-NA/null value and access the last observation directly
        last_observation = df_log.loc[df_log.first_valid_index()]

        # Extracting the scalar value from the Series
        last_obs_value = last_observation.iloc[0]

        # Adding the scalar value to the pred_cumsum Series
        pred_log = (last_obs_value + pred_cumsum)

        # Revert log transformation. Applying the exponential transformation
        pred_original = np.exp(pred_log)

        return pred_original


    def get_forecast_with_diff_arima(self, fitted_arima_model, df_log, step):
        forecast_result = fitted_arima_model.get_forecast(steps=step)
        forecast_values = forecast_result.predicted_mean

        forecast_cs = forecast_values.cumsum()  # cumulative sum to revert differencing

        last_observation = df_log.loc[df_log.last_valid_index()]

        # Extracting the scalar value from the Series
        last_obs_value = last_observation.iloc[0]

        forecast_log = (last_obs_value + forecast_cs)
        forecast = np.exp(forecast_log)
        
        return forecast
    
    def get_forecast_sarima(self, df, model_fit, step):
        # Create a list and append it with the predicted data
        prediction = []
        for i in range(step):
            yhat = model_fit.predict(len(df) + i-1)
            prediction.append([yhat.index[0], yhat[0]])
            # print(yhat)

        # print(yhat.index[0])
        # print(prediction)
        forecast = pd.DataFrame(prediction, columns = ['Date', 'Production'])

        return forecast

    def plot_prediction(self, df, pred_original, forecast):

        plt.plot(df, label='Original')
        plt.plot(pred_original, label='Predicted Original')
        plt.plot(forecast, color='red', label='Forecasted')
        plt.xlabel("Date")
        plt.ylabel("Production")
        plt.legend(loc='best')
        plt.show()

    def plot_forecast(self, df, forecast):

        plt.plot(df, label='Original')
        plt.plot(forecast, color='red', label='Forecasted')
        plt.xlabel("Date")
        plt.ylabel("Production")
        plt.legend(loc='best')
        plt.show()
    
    def get_concat_org_pred(self, df, forecast):
        df = df.reset_index()

        df_final = pd.concat([df, forecast], ignore_index=True, axis = 0)

        return df_final