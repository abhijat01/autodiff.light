import numpy as np


def to_one_hot(correct_cats, max_cat_num=None):
    r"""
    Given a vector or correct categories, it returns a 2D matrix
    containing one-hot encoded vectors
    It assumes that categories start from 0
    :param correct_cats: vector or 1D matrix containing correct categories
    :param max_cat_num:
    :return:
    """
    if max_cat_num is None:
        # Assume starting from 0
        max_value = np.max(correct_cats)
    else:
        max_value = max_cat_num

    one_hot = np.zeros((max_value + 1, correct_cats.size,))
    one_hot[correct_cats, np.arange(correct_cats.size)] = 1
    return one_hot
