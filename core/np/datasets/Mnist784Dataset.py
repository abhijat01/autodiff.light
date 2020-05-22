from sklearn.datasets import fetch_openml
import numpy as np
from core.np.utils import to_one_hot, LocalDataCache
from core import info
import os


class Mnist784:
    def __init__(self,  train_fraction=0.7, load_cache=True):
        self.train_fraction = train_fraction
        if not load_cache:
            self.data, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            self.targets = [int(category) for category in y]
        else:
            self.load_data_from_cache()

        self.targets = np.array(self.targets)
        self.training_set_indexes = None
        self.test_set_indexes = None
        self.max_cat_num = np.max(self.targets)
        self.split_test_train()

    def load_data_from_cache(self):
        local_cache = LocalDataCache()
        directory = local_cache.get_subdir("mnist")
        file_name = os.path.join(directory, "data784.npy")
        file_name = os.path.abspath(file_name)
        info("lading data array from:{}".format(file_name))
        self.data = np.load(file_name)

        file_name = os.path.join(directory, "target784.npy")
        file_name = os.path.abspath(file_name)
        info("Loading targets array from:{}".format(file_name))
        self.targets = np.load(file_name)

    def save(self):
        local_cache = LocalDataCache()
        save_dir = local_cache.get_subdir("mnist")
        file_name = os.path.join(save_dir, "data784")
        file_name = os.path.abspath(file_name)
        np.save(file_name, self.data)
        info("Saved data array into:{}".format(file_name))

        file_name = os.path.join(save_dir, "target784")
        file_name = os.path.abspath(file_name)
        np.save(file_name, self.targets)
        info("Saved targets array into:{}".format(file_name))

    def split_test_train(self):
        num_data_points = len(self.targets)
        num_train = int(num_data_points * self.train_fraction)
        info("Total number of data points:{}, number of training points:{}".format(num_data_points, num_train))
        self.training_set_indexes = np.random.choice(range(num_data_points), num_train, replace=False)
        self.test_set_indexes = set(range(num_data_points)) - set(self.training_set_indexes)
        self.test_set_indexes = list(self.test_set_indexes)

    def train_iterator(self, epochs, batch_size=1, one_hot=True):
        r"""
        Will by default return one_hot encoded vectors
        :param epochs:
        :param batch_size:
        :param one_hot:
        :return:
        """
        if batch_size >= len(self.training_set_indexes):
            raise Exception("Batch size {} too large for train list of size:{}".format(batch_size, len(self.training_set_indexes)))
        for epoch in range(epochs):
            row_indexes = np.random.choice(self.training_set_indexes, batch_size, replace=False)
            yield self.make_data(row_indexes, batch_size, one_hot)

    def test_iterator(self, num_tests, one_hot=True):
        for count in range(num_tests):
            row_indexes = np.random.choice(self.test_set_indexes, 1, replace=False)
            yield self.make_data(row_indexes, 1, one_hot)

    def make_data(self, row_indexes, batch_size, one_hot=True):
        x = np.zeros((self.data.shape[1], batch_size))
        for j, row_index in enumerate(row_indexes):
            x[:, j] = self.data[row_index].T
        x_targets = self.targets[row_indexes]
        if one_hot:
            y = to_one_hot(x_targets, self.max_cat_num)
        else:
            y = x_targets
        return x, y
