import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, median_absolute_error
from DataPreprocessing import *
from dataVisualize import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modellingPrep import *
from modelling import *
from regression import *
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from lstm_model import *
pd.set_option('display.max_columns',None)
pd.set_option('display.precision',2)

pd.set_option('float_format', '{:.2f}'.format)

np.set_printoptions(precision=2,  # limit to two decimal places in printing numpy
                    suppress=True,  # suppress scientific notation in printing numpy
                   )

from toolbox import *

df =""
clean_df=""
daily_df= ""
train_daily=""
test_daily=""
station_code = 'A010'
model = ""
def reverseTransform(prediction,y_train,y_test,title="Prediction"):

    y_reversed = []
    T = len(y_train)
    seasonality = 365
    for i in range(0,len(y_test)):
        if i <seasonality:
            y_reversed.append(prediction[i] + y_train[-seasonality+i])
        else:
            k = i-seasonality
            y_reversed.append(prediction[i] + y_reversed[k])
    forecasted_values = pd.Series(y_reversed)
    forecasted_values.index = prediction.index

    forecasted_values.plot( label='Forecast')
    y_test.plot( label='Actual Test Data')
    plt.title('Predictions Versus Test')
    plt.legend(loc= 'upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.plot(y_train.index, y_train.values,label ='Train')
    plt.plot(forecasted_values.index, forecasted_values.values, label='Forecast')
    plt.plot(y_test.index, y_test.values, label='Actual Test Data')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return forecasted_values
def forecastFunction(y,test,train,title):
    """
    :param y: differenced series
    :param test: test set (Relative Humidity)
    :param train: Train set (Relative Humidity)
    :return:
    """
    T = len(y) - 1
    test_len = len(test)
    y_hat = [0]
    y_hat.append(0.73 * y[T] + 0.041 * y[T - 5] + 0.056 * y[T-6])   # 1st step
    y_hat.append(0.73 * y_hat[1] + 0.041 * y[T - 4] + 0.056 * y[T-5]) # 2nd step
    y_hat.append(0.73 * y_hat[2] + 0.041 * y[T - 3] + 0.056 * y[T-4])  # 3rd step
    y_hat.append(0.73 * y_hat[3] + 0.041 * y[T - 2] + 0.056 * y[T-3])  # 4th step
    y_hat.append(0.73 * y_hat[4] + 0.041 * y[T - 1] + 0.056 * y[T-2])  # 5th step
    y_hat.append(0.73 * y_hat[5] + 0.041 * y[T - 0] + 0.056 * y[T-1])  # 6step
    y_hat.append(0.73 * y_hat[6] + 0.041 * y_hat[1] + 0.056 * y[T]) # 7th step
    for h in range(8, test_len+1):
        y_hat.append(0.73 * y_hat[h-1] + 0.041 * y_hat[h-6] + 0.056 * y_hat[h-7])  # 8 - last step predictions

    y_hat = y_hat[1:]
    forecasted = pd.Series(y_hat)
    forecasted.index = test.index
    reverseTransform(forecasted, train, test,title)


def preProcess(filepath):
    global df
    global station_code
    df = readDataFile(filepath,station_code)
    # data cleaning
    df = fixMissingValues(df)
    print(f"{'=' * 10}Done with filling missing values{'=' * 10}")
    print(df.describe())
    #from the description we can see that
    #doesn't rain much
    #can be an outlier in datset
    return df

def dataVisualization():
    global df
    global clean_df
    # sending temp and humidity at last for better visualization
    df.insert(len(df.columns) - 1, 'temp(C)', df.pop('temp(C)'))
    df.insert(len(df.columns) - 1, 'hmdy(%)', df.pop('hmdy(%)'))
    #df = df.loc["2006":"2010"]
    plotCorrelation(df)
    # we can see here
    # temp and humidity are dependent
    # depends on wind gust speed
    # humidity depends on dmin,dmax,dew
    # temp and humidity depends on solar radiation

    # hence more likely to focus first on
    # wind gust speed, dmin,dmax,dew,temp,humidity,tmin,tmax
    # print(df.head())
    clean_df = df
    plotDaily(clean_df,'hmdy(%)','Average Daily humidity over the years')

def main():
    global df
    global clean_df
    global daily_df
    global train_daily
    global test_daily
    global model
    filepath = 'Dataset/archive/north.csv'
    df = preProcess(filepath)
    dataVisualization()
    #since we have huge dataset will resample and find out humidity values by
    #resampling it for daily dataset
    daily_df = clean_df.resample('D').mean()


    print(f"{'='*5}Checking stationarity for Humidity")
    stationarityCheck(daily_df['hmdy(%)'], "Relative Humidity")
    #checkBasics(daily_df['hmdy(%)'], "Humidity")

    #daily_df.to_csv('clean_data.csv', index=True)

    print(f"{'=' * 5}Train Test Split")
    train_daily,test_daily = splitTrainTest(daily_df)
    seasonality = 365

    seas, trend, resi = decomposition(train_daily['hmdy(%)'], period=365)
    adjustTrend(train_daily['hmdy(%)'], trend, title="Detrended")
    adjustSeasonal(train_daily['hmdy(%)'], seas, title="Humidity")


    baseForecast(train_daily['hmdy(%)'], test_daily['hmdy(%)'], seasonality,title="Humidity(%)")
    simpleExponential(train_daily['hmdy(%)'][0], train_daily['hmdy(%)'], test_daily['hmdy(%)'], alpha=0.1, title="Relative Humidity(%)",
                      axes=None)
    print(f"Holt Winter Forecast for Humidity")
    holtWinterHumidity(seasonality, train_daily['hmdy(%)'], test_daily['hmdy(%)'], title="Holt Winter Prediction Humidity(%)")

    train_orig = train_daily.copy()
    test_orig = test_daily.copy()
    # #regression for humidity
    linearRegressionModel(train_daily,test_daily)

    train_daily = train_orig.copy()
    test_daily = test_orig.copy()
    ACF_PACF_Plot(train_daily['hmdy(%)'], 365 * 5)
    #
    df_365 = train_daily['hmdy(%)'].diff(periods=365)
    df_365 = removeNone(df_365)
    #
    stationarityCheck(df_365, title="Staionarity check Seasonal(365 diff)")
    ACF_PACF_Plot(df_365, 100)
    ry = autoCorrelationFunction(df_365, 50, title="Seasonal Differencing(365)", axes=None)

    calcGPAC(ry, 7, 7)

    #order(1,0)
    model_hat, e, Q, model = lm_predict(df_365, 1, 0, lags=100)
    ACF_PACF_Plot(e, 100)
    ry = autoCorrelationFunction(e, 50, title="Seasonal Differencing(365)", axes=None)
    calcGPAC(ry, 10, 10)


    model_hat, e, Q, model = lm_predict(df_365, 7, 0, lags=20)
    ry = autoCorrelationFunction(e, 50, title="Residual after 365 differencing(3,1)", axes=None)
    #calcGPAC(ry, 10, 10)


    #finding coefficients
    na = 7
    nb = 0
    theta = np.zeros(na + nb)
    y = df_365.copy()
    e_theta = generate_e_theta(y, theta, na, nb)
    theta_new, cov_theta, sigma_sq = lm(y, e_theta, theta, na, nb)
    #
    print(f"Esimated coefficient for order(7,0) \n{theta_new}")
    conf_interval, est_ar, est_ma = confidenceInterval(theta_new, cov_theta, na, nb)
    print(f"Roots are : {np.roots(est_ar)}")
    print(f"Standard deviation : {sigma_sq}")
    row =7
    col =7
    for r in range(0, row):
        print("[", end=" ")
        for c in range(0, col):
            print(cov_theta[r][c], end=" ")
        print("]")

    prediction = model.forecast(len(test_daily))
    forecastFunction(y, test_daily['hmdy(%)'], train_daily['hmdy(%)'],title = "Prediction using Original Forecast")
    forecasted_values = reverseTransform(prediction, train_daily['hmdy(%)'], test_daily['hmdy(%)'],title = "Prediction using the Fitted Equation")
    mse_arma_7_0 = mean_squared_error(test_daily['hmdy(%)'], forecasted_values)
    print(f"ARMA Model(7,0)")
    print(f"Mean Square error forecasted  {mse_arma_7_0:.2f}")
    print(f"Root Mean Square error forecasted {np.sqrt(mse_arma_7_0):.2f}")
    print(f"Roots of the polynomial are :")
    print(f"num : {np.real(np.roots(est_ma))}")
    print(f"den : {np.real(np.roots(est_ar))}")
    lstmMultiVariate(daily_df,daily_df['hmdy(%)'])

if __name__ == '__main__':
    main()




