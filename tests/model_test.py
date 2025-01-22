import pytest
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def generate_fake_data(n=100):
    # Генерация случайных данных
    return np.random.rand(n)


def test_train_lstm_model():
    data = generate_fake_data()

    # Проверяем, что функция не вызывает ошибок
    predictions, train_size, time_step = train_lstm_model(data)

    # Проверяем размеры данных
    assert len(predictions) == len(data) - int(len(data) * 0.8) - time_step - 1, \

    # Проверяем, что время шага корректно
    assert time_step == 60, "time_step должен быть 60."

