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

    def test_file_persistence(self):
        w1, b1, w2, b2, w3, b3 = self._hard_coded_arrays()

        saver = utils.FilePersistenceHelper("iris.multilayer")

        saver.push_level("layer1")
        saver.set_numpy_array(w1, "w")
        saver.set_numpy_array(b1, "b")
        saver.pop_level()

        saver.push_level("layer2")
        saver.set_numpy_array(w2, "w")
        saver.set_numpy_array(b2, "b")
        saver.pop_level()

        saver.push_level("layer3")
        saver.set_numpy_array(w3, "w")
        saver.set_numpy_array(b3, "b")
        saver.save_to_file()

        reader = utils.FilePersistenceHelper.read_from_file("iris.multilayer")

        reader.push_level("layer1")
        w = reader.get_numpy_array("w")
        np.testing.assert_equal(w1, w)
        b = reader.get_numpy_array("b")
        np.testing.assert_equal(b1, b)
        reader.pop_level()

        reader.push_level("layer2")
        w = reader.get_numpy_array("w")
        np.testing.assert_equal(w2, w)
        b = reader.get_numpy_array("b")
        np.testing.assert_equal(b2, b)
        reader.pop_level()

        reader.push_level("layer3")
        w = reader.get_numpy_array("w")
        np.testing.assert_equal(w3, w)
        b = reader.get_numpy_array("b")
        np.testing.assert_equal(b3, b)
        reader.pop_level()

    def _hard_coded_arrays(self):
        w1 = np.array([[0.3648, 0.4178, 0.1107, 0.3528],
                       [-0.1352, 0.1000, -0.2287, -0.3533],
                       [-0.0444, -0.2728, -0.2197, -0.4922],
                       [0.1953, 0.1613, -0.1723, -0.4593],
                       [-0.4630, -0.1699, 0.3542, -0.3734],
                       [0.2785, 0.3824, 0.2087, -0.1753],
                       [-0.2305, -0.0334, -0.3379, 0.1667],
                       [0.3676, -0.4983, -0.4050, 0.0118],
                       [-0.3470, -0.3068, 0.1592, -0.2517],
                       [-0.3822, -0.1634, -0.0815, -0.2605],
                       [-0.2786, 0.3482, -0.4424, -0.4211],
                       [0.2079, -0.4462, 0.3021, -0.2924],
                       [-0.4455, 0.1701, -0.4978, 0.2291],
                       [-0.2363, -0.4727, 0.4679, 0.3664],
                       [0.0356, -0.4728, 0.4352, 0.1298],
                       [0.3458, -0.1421, 0.4326, -0.4289]])
        b1 = np.array([-0.2420, -0.3605, 0.1565, 0.1164, -0.2386, -0.4217, -0.0773, -0.0902,
                       -0.3399, 0.2479, -0.1368, 0.1328, -0.0935, 0.3034, -0.1878, -0.4137])
        w2 = np.array([[0.0179, 0.2245, 0.0410, 0.1400, 0.1753, -0.0133, 0.2359, 0.1599,
                        -0.1014, 0.0888, 0.0336, 0.1069, 0.2021, -0.2078, 0.1241, -0.2466],
                       [-0.0831, 0.2404, 0.1575, 0.1075, 0.0297, 0.1771, -0.1952, -0.2346,
                        -0.1269, 0.0890, -0.0778, -0.0041, 0.1113, -0.1204, -0.0024, 0.1994],
                       [0.2101, -0.0595, -0.2110, 0.1711, 0.1235, 0.0181, -0.1251, 0.1947,
                        0.2219, -0.1931, -0.0980, -0.1768, -0.1196, 0.0056, 0.1109, -0.0292],
                       [0.2365, -0.0695, 0.0129, -0.0142, -0.1262, 0.1906, -0.1356, 0.0342,
                        0.2492, 0.1121, 0.2474, -0.1357, 0.1559, -0.1849, -0.0596, 0.2322],
                       [0.0866, 0.2246, 0.1975, -0.0982, -0.1896, 0.0823, 0.0927, 0.2446,
                        -0.2303, 0.2308, 0.0024, 0.0124, -0.0352, -0.1604, -0.0090, -0.0086],
                       [-0.2172, -0.0150, -0.2309, 0.1886, -0.0721, -0.2308, 0.1307, 0.1452,
                        0.0108, 0.1264, -0.0307, -0.1473, -0.1139, -0.1358, -0.0381, 0.1257],
                       [-0.2021, 0.1673, -0.1039, -0.1478, -0.1899, 0.0728, 0.2291, 0.1776,
                        -0.1111, -0.2363, -0.1667, 0.1284, 0.0321, 0.0025, -0.1172, -0.2465],
                       [-0.1580, 0.0926, 0.1701, -0.1008, -0.1641, -0.2080, -0.1524, -0.1190,
                        -0.0743, 0.0564, 0.1939, 0.1880, 0.1123, 0.1196, -0.2310, -0.0793],
                       [0.1826, 0.0572, -0.1104, 0.1376, 0.0184, 0.1704, -0.2070, 0.0882,
                        0.2335, -0.0041, -0.0163, -0.0149, -0.1895, -0.1003, 0.1097, -0.1816],
                       [-0.1266, 0.1819, 0.2048, 0.2406, 0.2037, 0.1301, 0.0120, 0.0957,
                        0.0296, 0.0433, -0.0663, 0.1723, 0.0152, 0.0220, -0.2107, 0.0548]])
        b2 = np.array([-0.0275, -0.1463, -0.0197, 0.0397, -0.1154, -0.2203, -0.1066, -0.0064,
                       0.1709, 0.1595])
        w3 = np.array([[-0.0612, 0.2616, -0.2794, -0.0259, 0.1405, -0.0295, -0.1572, -0.2135,
                        0.1265, 0.0025],
                       [-0.2529, -0.0696, -0.2372, -0.1703, -0.0103, 0.0861, -0.0221, 0.0197,
                        0.2440, 0.1436],
                       [-0.2652, -0.1656, 0.1708, 0.0962, -0.1399, -0.2748, 0.0443, 0.2490,
                        0.2023, 0.0357]])
        b3 = np.array([-0.1787, -0.2090, -0.0768])

        return w1, b1, w2, b2, w3, b3


if __name__ == '__main__':
    unittest.main()
