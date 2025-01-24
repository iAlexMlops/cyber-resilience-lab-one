import warnings

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def create_train_test(input_data, time_step=60):
    data = input_data.reshape(-1, 1)

    # Приводим данные к диапазону от 0 до 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Создаем тренировочный и тестовый наборы данных
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Изменяем форму данных для входа в LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler


def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


def create_model(time_step=60):
    # Создаем модель LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    return model


if '__main__' == __name__:
    df = pd.read_csv('data/portfolio_data.csv')
    X_train, y_train, X_test, y_test, scaler = create_train_test(df['DPZ'].values, time_step=60)

    model = create_model(time_step=60)
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    print(model.summary())
