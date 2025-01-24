import unittest
import numpy as np
from keras.models import Sequential

from LSTM import create_dataset, create_model


class TestLSTMFunctions(unittest.TestCase):
    def setUp(self):
        # Метод setUp выполняется перед каждым тестом
        self.sample_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_create_dataset(self):
        time_step = 3
        X, y = create_dataset(self.sample_data.reshape(-1, 1), time_step)

        # Проверяем размеры
        self.assertEqual(X.shape, (6,3))
        self.assertEqual(y.shape, (6,))

        # Проверяем значения
        expected_X = np.array([[1, 2, 3],
                               [2, 3, 4],
                               [3, 4, 5],
                               [4, 5, 6],
                               [5, 6, 7],
                               [6, 7, 8]])
        expected_y = np.array([4, 5, 6, 7, 8, 9])

        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

    def test_create_model(self):
        model = create_model(time_step=3)
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 4)  # 2 LSTM + 2 Dense

    def test_model_compilation(self):
        model = create_model(time_step=3)
        # Проверяем, корректно ли собрана модель
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)


if __name__ == '__main__':
    unittest.main()
