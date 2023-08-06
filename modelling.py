import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from toolbox import *


#calculateErrorCheck(test,forecast,title = "Holt Winter")
def calculateErrorCheck(test,prediction,title="error"):
    forecast_error = test - prediction
    mean_sqr_fore_error = np.mean(forecast_error ** 2)
    rms = math.sqrt(mean_sqr_fore_error)
    #autoCorrelation(mean_error,365*3,"forecast error", show_plot=True)
    print(f"Forecasted Error")
    print(title)
    print(f"Mean square error for {title} is : {mean_sqr_fore_error:.2f}")
    print(f"Root Mean Square Error {title} is : {rms:.2f}")

    #print(f" Mean Residual error : {np.mean(forecast_error)} \nVariance Residual Error :{np.var(forecast_error)}")
    print(f" Mean Forecast Error : {np.mean(forecast_error)} \nVariance Forecast Error:{np.var(forecast_error)}")

def calculateErrorCheckResidual(train,onestep,title="error"):
    residualError = train - onestep
    mean_sqr_resid_error = np.mean(residualError ** 2)
    rms = math.sqrt(mean_sqr_resid_error)
    #autoCorrelation(mean_error,365*3,"forecast error", show_plot=True)

    print(f"mean square error for {title} is : {mean_sqr_resid_error}")
    print(f"Root Mean Square Error {title} is : {rms}")

    #print(f" Mean Residual error : {np.mean(forecast_error)} \nVariance Residual Error :{np.var(forecast_error)}")
    print(f" Mean Forecast Error : {np.mean(rms)} \nVariance Forecast Error:{np.var(rms)}")
