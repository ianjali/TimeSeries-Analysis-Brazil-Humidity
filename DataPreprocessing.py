import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, median_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

import folium
from statsmodels.tsa.seasonal import STL
pd.set_option('display.max_columns',None)
pd.set_option('display.precision',2)
renamed_columns = ['data','hora','precipitacao total,horario (mm)','pressao atmosferica ao nivel da estacao (mb)',
                   'pressao atmosferica max. na hora ant. (aut) (mb)','pressao atmosferica min. na hora ant. (aut) (mb)','radiation (kj/m2)',
                   'temperatura do ar - bulbo seco (°c)','temperatura do ponto de orvalho (°c)','temperatura maxima na hora ant. (aut) (°c)',
                   'temperatura minima na hora ant. (aut) (°c)','temperatura orvalho max. na hora ant. (aut) (°c)',
                   'temperatura orvalho min. na hora ant. (aut) (°c)','umidade rel. max. na hora ant. (aut) (%)','umidade rel. min. na hora ant. (aut) (%)','umidade relativa do ar, horaria (%)','vento direcao horaria (gr) (° (gr))','vento rajada maxima (m/s)','vento velocidade horaria (m/s)','region','state','station','station_code','latitude','longitude','height']
renamed_columns_en = ['date','hour','total precipitation (mm)','pressao atmosferica ao nivel da estacao (mb)',
                      'atmospheric pressure max. in the previous hour (mb)','atmospheric pressure min. in the previous hour (mb)',
                      'radiation (kj/m2)','air temperature - dry bulb (°c)','dew point temperature (°c)','max. temperature in the previous hour (°c)',
                      'min. temperature in the previous hour (°c)','dew temperature max. in the previous hour (°c)',
                      'dew temperature min. in the previous hour (°c)','relative humidity max. in the previous hour (%)',
                      'relative humidity min. in the previous hour (%)','air relative humidity (%)','wind direction (° (gr))',
                      'wind rajada maxima (m/s)','wind speed (m/s)','region','state','station','station_code','latitude','longitude','height']
abbreviation = ['date','hour','prcp(mm)', 'atmp(mb)', 'atmmax', 'atmmin','radi(KJ/m2)','temp(C)','dewp(C)','tmax','tmin','dmax','dmin','hmax(%)','hmin','hmdy(%)',
                'wdir(deg)', 'wgust(m/s)', 'wdsp(m/s)', 'regi','prov','snam','inme','lat','lon','elvt']
def interpolateSeasonalDecomposition(df,interpolate_df,var='hmdy(%)'):
    stl = STL(df[var], period=24)
    result = stl.fit()



    # extract the components
    seasonal = result.seasonal
    trend = result.trend
    residual = result.resid

    # Estimate missing trend values using Simple Exponential Smoothing
    trend =trend.fillna(value=SimpleExpSmoothing(trend).fit().fittedvalues, inplace=True)
    # Check for missing values in the trend component
    if trend.isnull().values.any():
        trend = trend.interpolate(method='spline', limit_direction='both')
    # Estimate missing seasonal values using seasonal interpolation
    seasonal = seasonal.interpolate(method='spline', limit_direction='both')

    # Estimate missing residual values using ARIMA
    residual = residual.fillna(value=ARIMA(residual, order=(1, 0, 1)).fit().fittedvalues, inplace=True)

    # Combine the estimated components to obtain a complete time series
    filled_data = trend+ seasonal + residual

    # Fill missing values using estimated components
    #filled_data = trend + seasonal + residual
    #filled_data[df[var].isnull()] = df[var][df[var].isnull()]
    # Replace missing values with filled data
    #filled_data[df.isnull()] = df[df.isnull()].index.map(lambda x: seasonal[x.day - 1])
    # Plot original and filled data
    plt.plot(df[var], label='Original Data')
    plt.plot(filled_data, label='Filled Data')
    plt.legend()
    plt.show()
    #df[var] = filled_data
    return  df
def fixMissingValues(df):
    df = df.replace(to_replace=-9999.000000, value=np.NaN)

    interpolate_df = df.interpolate(option='spline')
    #df = interpolateSeasonalDecomposition(df,interpolate_df)
    df = interpolate_df


    return df
def plotMap(map_df,station_code='A010'):
    """
    plot the map of given datframe
    :param map_df: lat,lon,station code columns
    :return:
    """
    #print(map_df)
    map = folium.Map(location=[map_df.lat.mean(), map_df.lon.mean()],
                     zoom_start=14, control_scale=True)
    for index, location_info in map_df.iterrows():
        if location_info["inme"] == station_code:
            icon = folium.Icon(color='red')
            folium.Marker([location_info['lat'], location_info['lon']], popup=location_info["inme"], icon=icon).add_to(map)
        else :
            folium.Marker([location_info['lat'], location_info['lon']], popup=location_info["inme"]).add_to(map)
    map.save('station_north.html')
    #return map
    print(f"{'='*10}Done with mapping{'='*10}")
def readDataFile(path,station_code):
    """
    This function reads data in a time series format
    :param path: path to the file you want to read
    :return: dataframe
    """
    df = pd.read_csv(path,header=0,parse_dates=[0],index_col=0)
    #df.drop(['index'], inplace=True, axis=1)
    df.columns = abbreviation
    df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['hour'])
    #plot map of stations
    map_df = df[['lat','lon','inme']].drop_duplicates()
    plotMap(map_df)
    #removing date,hour,region,prov,station name,latitude,longitude,elevation
    df.drop(['date', 'hour', 'regi', 'prov', 'snam', 'lat', 'lon', 'elvt'], inplace=True, axis=1)

    #choosing that particular station only
    df = df[df['inme']==station_code]
    df.set_index(['date_time'],inplace=True)
    return df


