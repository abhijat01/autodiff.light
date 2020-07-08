import numpy as np
import json
import os


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


class LocalDataCache:
    def __init__(self, cache_dir=None):
        self.data_dir = cache_dir
        if self.data_dir is None:
            self.data_dir = os.path.expanduser("~")
            self.data_dir = os.path.join(self.data_dir, ".autodiff.light")
            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)
                if not os.path.isdir(self.data_dir):
                    raise Exception("Could not create data dir:{}".format(self.data_dir))

    def get_subdir(self, data_name):
        data_dir = os.path.join(self.data_dir, data_name)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        return data_dir


def always_filter(dir, name):
    return True


def ext_filter(ends_with_string):
    def f(dir, name):
        return name.endswith(ends_with_string)
    return f


def get_file_list(directory: str = ".", return_full_path: bool = True, files_only: bool = True,
                  filename_filter=always_filter):
    """

    :param directory:
    :param return_full_path: if false, only the file name will be returned, otherwise complete path W.R.T to directory
    will be returned.
    :param files_only: This setting will always override filter hence filter will
    never get to see this a directory if this is set to true
    :param filename_filter: should accept two arguments, the directory and the filename and return True if this file should be
    included in the listing
    :return:
    """
    file_list = os.listdir(directory)
    ret_list = []
    for file_name in file_list:
        full_path = os.path.join(directory, file_name)
        if files_only:
            if not os.path.isdir(full_path):
                if filename_filter(directory, file_name):
                    if return_full_path:
                        ret_list.append(full_path)
                    else:
                        ret_list.append(file_name)

    return ret_list

