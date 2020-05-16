import numpy as np
import sklearn
from sklearn import datasets
from core.np.utils import to_one_hot
from core import debug


class Iris:
    def __init__(self, split_now=True, train_fraction=.7):
        self.iris = datasets.load_iris()
        self.data = self.iris.data
        self.targets = self.iris.target
        self.max_cat_num = np.max(self.targets)
        self.train_idx = None
        self.test_idx = None
        self.train_fraction = train_fraction
        if split_now:
            self.split_test_train()

    def split_test_train(self):
        num_data_points = len(self.targets)
        num_train = int(num_data_points * self.train_fraction)
        self.train_idx = np.random.choice(range(num_data_points), num_train, replace=False)
        self.test_idx = set(range(num_data_points)) - set(self.train_idx)
        self.test_idx = list(self.test_idx)

    def train_iterator(self, epochs, batch_size=1, one_hot=True):
        r"""
        Will by default return one_hot encoded vectors 
        :param epochs:
        :param batch_size:
        :param one_hot:
        :return:
        """
        if batch_size >= len(self.train_idx):
            raise Exception("Batch size {} too large for train list of size:{}".format(batch_size, len(self.train_idx)))

        for epoch in range(epochs):
            row_indexes = np.random.choice(self.train_idx, batch_size, replace=False)
            x = np.zeros((self.data.shape[1], batch_size))
            for j in range(len(row_indexes)):
                x[:, j] = self.data[j, :].T
            x_targets = self.targets[row_indexes]
            if one_hot:
                y = to_one_hot(x_targets, self.max_cat_num)
            else:
                y = x_targets
            #debug("x_targets = np.{}".format(repr(x_targets)))
            yield x, y

    def test_iterator(self, num_tests, one_hot=True):
        for count in range(num_tests):
            row_indexes = np.random.choice(self.test_idx, 1, replace=False)
            x = np.zeros((self.data.shape[1], 1))
            for j in range(len(row_indexes)):
                x[:, j] = self.data[j, :].T
            x_targets = self.targets[row_indexes]
            if one_hot:
                y = to_one_hot(x_targets, self.max_cat_num)
            else:
                y = x_targets
            #debug("x_targets = np.{}".format(repr(x_targets)))
            yield x, y



