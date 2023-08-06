import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_north = pd.read_csv('Dataset/north.csv')
df_a10 = df_north[df_north['station_code']=='A010']
df_a10.rename(columns = {'Data':'Date',
                         'Hora' : 'Hour',
                         'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)' : 'Precipitation_last_hr(mm)',
                         'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)':'Atm Pressure(mb)',
                         'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)':'Max Air Pressure(mb)',
                         'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)':'Min Air Pressure(mb)',
                         'RADIACAO GLOBAL (Kj/m²)':'Solar Radiation(KJ/M2)',
                        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)':'Temp(C)',
                        'TEMPERATURA DO PONTO DE ORVALHO (°C)':'Dew Point temp instant(C)',
                        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)' : 'Max Temp last hr(C)',
                        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Min Temp last hr(C)',
                        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)' : 'Max Dew Point Temp last hr(C)',
                        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'Min Dew Point Temp last hr(C)',
                        'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)' : 'Max Relative Humid Temp last hr(%)',
                        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)' : 'Min Relative Humid Temp last hr(%)' ,
                        'UMIDADE RELATIVA DO AR, HORARIA (%)' : 'Humidity(%)',
                        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))' : 'Wind Direction(radius degree 0-360)',
                        'VENTO, RAJADA MAXIMA (m/s)':'Wind Gust(m/sec)',
                        'VENTO, VELOCIDADE HORARIA (m/s)':'Wind Speed(m/sec)'  },inplace = True)

df_a10 = df_a10.drop(['region','station','state','latitude','longitude','height','station_code'],axis=1)
# converting dates/time columns into a datetime object
df_a10["Date"] = pd.to_datetime(df_a10["Date"])
# set the new datetime column as the index
df_a10 = df_a10.set_index("Date")
df_a10["year"] = df_a10.index.year
df_a10["month"] = df_a10.index.month
df_a10["Day"] = df_a10.index.day

#removing the outliers/missing values
df = df_a10[df_a10["Temp(C)"] != -9999.000000]
x = df.groupby(["year", "month"])["Temp(C)"].mean()
df_wide = x.unstack()
print(df_wide)
#df_wide

#df_clean[df_clean["Air Temp(instant) (C)"] != -9999.000000]["Air Temp(instant) (C)"].describe()
df = df[df["Humidity(%)"] != -9999.000000]
print(df.describe())

#Relative humidity for all years
month_avg_hum=df["Humidity(%)"].resample("M").mean()
month_avg_hum.plot()
plt.ylabel("Humidity(%)")
plt.title("Relative Humidity(%) for all years")
plt.show()
#Temp for all years
month_avg_temp=df["Temp(C)"].resample("M").mean()
month_avg_temp.plot()
plt.ylabel("Temp(C)")
plt.title("Air Temp(instant) (C) for all years")
plt.show()
#Air temp for specified year
year = "2006 - 2007"
day_df =df.loc["2006":"2007"]
day_avg_temp=day_df["Temp(C)"].resample("D").mean()
day_avg_temp.plot()
plt.ylabel("Temp(C)")
title = "Air Temp(instant) (C) for "+year+" years"
plt.title(title)
plt.show()

#Humidity for specified year
day_avg_hum=day_df["Humidity(%)"].resample("D").mean()
day_avg_hum.plot()
plt.ylabel("Humidity(%)")
title = "Relative Humidity(% instant) for "+year+" years"
plt.title(title)
plt.show()

