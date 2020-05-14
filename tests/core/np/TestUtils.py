import unittest
import numpy as np
import core.np.utils as utils
from core import debug


class TestUtilFunctions(unittest.TestCase):
    def test_to_one_hot(self):
        one_hot_target = np.array([[0, 1, 0, 0], [0, 0, 0, 1]]).T
        debug("original one_hot_target = np.{}".format(repr(one_hot_target)))
        cats = np.array([1, 3])
        debug("cats = np.{}".format(repr(cats)))
        one_hot = utils.to_one_hot(cats)
        debug("one_hot = np.{}".format(repr(one_hot)))
        np.testing.assert_array_almost_equal(one_hot_target, one_hot)
        cats = np.array([3])
        one_hot = utils.to_one_hot(cats, 5)
        debug("one_hot = np.{}".format(repr(one_hot)))
        expected = np.array([[0, 0, 0, 1, 0, 0]]).T
        np.testing.assert_array_almost_equal(expected, one_hot)
        cats = np.array([3, 2])
        debug("cats = np.{}".format(repr(cats)))
        one_hot = utils.to_one_hot(cats, 7)
        debug("one_hot = np.{}".format(repr(one_hot)))
        expected = np.array([[0., 0.],
                             [0., 0.],
                             [0., 1.],
                             [1., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.]])
        np.testing.assert_array_almost_equal(expected, one_hot)


if __name__ == '__main__':
    unittest.main()
