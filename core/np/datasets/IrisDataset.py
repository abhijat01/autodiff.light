import numpy as np
from sklearn import datasets
from core.np.utils import to_one_hot


class Iris:
    def __init__(self, split_now=True, train_fraction=.7):
        self.iris = datasets.load_iris()
        self.data = self.iris.data
        self.targets = self.iris.target
        self.max_cat_num = np.max(self.targets)
        self.train_idx = None
        self.test_set_indexes = None
        self.train_fraction = train_fraction
        if split_now:
            self.split_test_train()

    def split_test_train(self):
        num_data_points = len(self.targets)
        num_train = int(num_data_points * self.train_fraction)
        self.train_idx = np.random.choice(range(num_data_points), num_train, replace=False)
        self.test_set_indexes = set(range(num_data_points)) - set(self.train_idx)
        self.test_set_indexes = list(self.test_set_indexes)

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
            yield self.make_data(row_indexes, batch_size, one_hot)

    def test_iterator(self, num_tests, one_hot=True):
        batch_size = 1
        for count in range(num_tests):
            row_indexes = np.random.choice(self.test_set_indexes, batch_size, replace=False)
            yield self.make_data(row_indexes, batch_size, one_hot)

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

