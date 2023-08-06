
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential
# from keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout  # ,CuDNNLSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
def lstmModel(y):
    train_size = int(len(y) * 0.8)  # 80/20 split
    train_data = y[:train_size]
    test_data = y[train_size:]

    # Scale the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    test_data = scaler.transform(test_data.values.reshape(-1, 1)).flatten()

    # Create input sequences and target values
    time_step = 365  # length of input sequence
    train_X, train_y = [], []
    test_X, test_y = [], []
    for i in range(len(train_data) - time_step):
        train_X.append(train_data[i:i+time_step])
        train_y.append(train_data[i+time_step])
    for i in range(len(test_data) - time_step):
        test_X.append(test_data[i:i+time_step])
        test_y.append(test_data[i+time_step])
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(time_step, 1), return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    history = model.fit(train_X, train_y,  batch_size=16, validation_split=.1, epochs=12, verbose=1)
    plt.figure()
    plt.plot(history.history['loss'], 'r', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.legend()
    plt.show()


    return model

def lstmMultiVariate(df,y):
    #df = df[['prcp(mm)', 'atmp(mb)', 'atmmax', 'atmmin', 'radi(KJ/m2)', 'dewp(C)','tmax', 'tmin', 'dmax', 'dmin', 'hmax(%)', 'hmin', 'wdir(deg)','wgust(m/s)', 'wdsp(m/s)', 'inme', 'temp(C)']] #df.drop(['hmdy(%)'], axis=1)
    df2 = df.copy()
    #df2 = df2.drop(['hmdy(%)'], axis=1)
    # col_to_move = 'temp'
    # last_col = df2.pop(col_to_move)
    # df2.insert(len(df2.columns), col_to_move, last_col)  # this is just for my
    # dataset, basically you need to use your original dataframe heres

    scaler = StandardScaler()
    scaler = scaler.fit(df2)
    scaled_data = scaler.transform(df2)
    train_size = int(len(df2) * 0.8)
    train_data = scaled_data[:train_size, :]
    time_step = 365
    train_X, train_y = [], []
    n_past = len(y) - train_size
    for i in range(n_past, len(train_data)):
        train_X.append(train_data[i - time_step:i, 0:train_data.shape[1] - 1])
        train_y.append(train_data[i, train_data.shape[1] - 1])

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    history = model.fit(train_X, train_y, batch_size=16, validation_split=.1, epochs=10, verbose=1)
    plt.figure()
    plt.plot(history.history['loss'], 'r', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.legend()
    plt.show()
    dataset = df2.values
    test_data = scaled_data[train_size - 365:, :]
    x_test = []
    y_test = dataset[train_size:, -1]

    for i in range(365, len(test_data)):
        x_test.append(test_data[i - 365:i, 0:train_data.shape[1] - 1])

    x_test = np.array(x_test)
    predictions = model.predict(x_test)
    forecast_copies = np.repeat(predictions, train_X.shape[2] + 1, axis=-1)
    predictions = scaler.inverse_transform(forecast_copies)[:, -1]

    train = df2.iloc[:train_size]
    valid = df2.iloc[train_size:]
    valid['Predictions'] = predictions

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Humidity prediction(LSTM)')
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel('Humidity')
    ax.plot(train['hmdy(%)'], 'blue')
    ax.plot(valid['hmdy(%)'], 'green')
    ax.plot(valid['Predictions'], 'orange')
    ax.legend(["Train", "Val", "Predictions"], loc="lower right", fontsize=18)
    ax.grid()
    plt.show()

    mse_lstm = mean_squared_error(valid['hmdy(%)'], valid['Predictions'])
    print("Mean Square Error for LSTM : ", mse_lstm)
    print("Root Mean Square Error for LSTM : ", np.sqrt(mse_lstm))
