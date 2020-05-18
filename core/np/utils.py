import numpy as np
import json


class FilePersistenceHelper:
    def __init__(self, start_part: str, version: int = 1, qualifier: str = None):
        self.root_name = start_part
        self.current_name = start_part
        self.root_dict = {start_part: {}}
        self.name_list = [start_part]
        self.sep = '.'
        self.version = str(version)
        self.qualifier = qualifier

    @staticmethod
    def _file_name(start_part: str, version: int = 1, qualifier: str = None):
        name = start_part + "." + str(version)
        if qualifier:
            name = name + "." + qualifier
        return name + ".json"

    def _get_file_name(self):
        return FilePersistenceHelper._file_name(self.root_name, self.version, self.qualifier)

    def _current_dict(self):
        dict_to_use = self.root_dict
        for name in self.name_list:
            if not (name in dict_to_use):
                dict_to_use[name] = {}
            dict_to_use = dict_to_use[name]
        return dict_to_use

    def push_level(self, level_name):
        if self.sep in level_name:
            raise Exception("Invalid level name:{}. Names cannot contain underscore".format(level_name))
        self.name_list.append(level_name)

    def pop_level(self):
        self.name_list.pop()

    def set_numpy_array(self, np_array, array_name)->None:
        current_dict = self._current_dict()
        current_dict[array_name] = np_array.tolist()

    def get_numpy_array(self, array_name) ->np.array:
        current_dict = self._current_dict()
        if array_name in current_dict:
            return np.array(current_dict[array_name])
        return None

    def save_to_file(self, fname: str = None) -> None:
        json_file = fname
        if not json_file:
            json_file = self._get_file_name()
        dict_to_save = self.root_dict
        with open(json_file, 'w') as jf:
            json.dump(dict_to_save, jf, indent=4, sort_keys=True)

    @staticmethod
    def read_from_file(start_name: str, version: int = 1, qualifier: str = None):
        json_file = FilePersistenceHelper._file_name(start_name, version, qualifier)
        with open(json_file, 'r') as jf:
            json_dict = json.load(jf)
        fph = FilePersistenceHelper(start_name, version, qualifier)
        fph.root_dict = json_dict
        return fph


def to_one_hot(correct_cats: np.array, max_cat_num=None) -> np.array:
    r"""
    Given a vector of correct categories, it returns a 2D matrix
    containing one-hot encoded vectors.


    This method assumes that categories start from 0.
    For example:


    cats =  np.array([3, 2])
    one_hot =  to_one_hot(cats, 7)
    one_hot_exp = np.array([[0., 0.],
       [0., 0.],
       [0., 1.],
       [1., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.]])
    np.testing.assert_array_almost_equal(one_hot_exp, one_hot)

    :param correct_cats: vector or 1D matrix containing correct categories
    :param max_cat_num: if set, it will not use max value as the max category number.
    Useful all categories  may not be present in the input (in particular the max
    category)
    :return: a 2D matrix of dimensions max_cat+1 X correct_cats.size
    """
    if max_cat_num is None:
        # Assume starting from 0
        max_value = np.max(correct_cats)
    else:
        max_value = max_cat_num

    one_hot = np.zeros((max_value + 1, correct_cats.size,))
    one_hot[correct_cats, np.arange(correct_cats.size)] = 1
    return one_hot
