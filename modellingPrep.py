from sklearn.model_selection import train_test_split
from toolbox import *
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from modelling import  *
def splitTrainTest(df):
    # Split the data into a training set and a testing set
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42,shuffle=False)
    return train_data,test_data


# def holtWinterTemp(s,train,test,title="Holt Winter Prediction"):
#     seasonal_period = s
#     model = ExponentialSmoothing(train, seasonal_periods=seasonal_period,
#                                  seasonal='add')
#                                  #trend='add', seasonal='add')  # , damped=True)
#     # Fit the model
#     model_fit = model.fit()
#
#     # Generate forecasts for a certain number of periods
#     num_periods = len(test)
#     forecast = model_fit.forecast(num_periods)
#     # Plot the training data and the forecast
#     plt.plot(train.index, train.values, label='Training Data')
#     plt.plot(test.index, test.values, label='Actual Test Data')
#     plt.plot(forecast.index, forecast.values, label='Forecast Using Mul')
#     plt.title(title)
#     plt.ylabel('Relative Humdity(%)')
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     calculateErrorCheck(test,forecast)
def holtWinterHumidity(s,train,test,title="Holt Winter Prediction"):
    seasonal_period = s
    model = ExponentialSmoothing(train, seasonal_periods=seasonal_period,
                                 seasonal='add')
                                 #trend='add', seasonal='add')  # , damped=True)
    # Fit the model
    model_fit = model.fit()

    # Generate forecasts for a certain number of periods
    num_periods = len(test)
    forecast = model_fit.forecast(num_periods)
    # Plot the training data and the forecast
    plt.plot(train.index, train.values, label='Training Data')
    plt.plot(test.index, test.values, label='Actual Test Data')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.ylabel('Relative Humidity(%)')
    plt.grid()
    plt.tight_layout()
    plt.show()

    calculateErrorCheck(test,forecast,title = "Holt Winter")


def baseForecast(train,test,s,title="Prediction"):
    flatForecast(train, test, method="Naive", title=title)
    flatForecast(train, test, method="Average", title=title)
    flatForecast(train, test, method="Drift", title=title)

def plotAllAutoCorrelation(series):
    #autoCorrelation(series, 4*24, title="4 days", show_plot=True)

    daily_average_df = series.resample('D').mean()
    autoCorrelation(daily_average_df, 30*12*3, title="3 Years", show_plot=True)

    #autoCorrelation(daily_average_df, 30 * 24, title="2 years(Daily Average) ", show_plot=True)

    #monthly_average_df = series.resample('M').mean()
    #autoCorrelation(monthly_average_df, 12*2, title="2 Years(Monthly Average) ", show_plot=True)

    #yearly_average_df = series.resample('Y').mean()
    #autoCorrelation(yearly_average_df, 10, title="10 Years(Yearly Average) ", show_plot=True)