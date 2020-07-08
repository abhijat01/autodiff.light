import numpy as np
from core import log_at_info, debug
import core.np.utils as u
import os
import unicodedata
import string
import random


class NameDS:
    def __init__(self, data_dir):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        self.data_dir = data_dir
        self.file_list = u.get_file_list(self.data_dir, u.ext_filter(".txt"))
        self.category_files = {}
        self.categories_list = []
        self.data = {}
        self.total_names = 0
        for file_name in self.file_list:
            language = os.path.splitext(os.path.basename(file_name))[0]
            self.category_files[language] = file_name
            lines = self.read_lines(file_name)
            self.data[language] = lines
            self.total_names += len(lines)
            self.categories_list.append(language)
        self.n_categories = len(self.category_files)
        debug("[NameDS.__init__()] self.total_names = {}".format(self.total_names))

    def read_lines(self, filename):
        with open(filename, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def letter_to_index(self, letter):
        return self.all_letters.find(letter)

    def line_to_numpy(self, line):
        line_array = np.zeros((self.n_letters, 1, len(line)))
        for li, letter in enumerate(line):
            line_array[self.letter_to_index(letter)][0][li] = 1
        return line_array

    def random_choice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def random_training_example(self):
        category = self.random_choice(self.categories_list)
        line = self.random_choice(self.data[category])
        category_idx = self.categories_list.index(category)
        line_tensor = self.line_to_numpy(line)
        return category, line, category_idx, line_tensor

    def random_training_batch(self, batch_size=0):

        category = self.random_choice(self.categories_list)
        line = self.random_choice(self.data[category])
        category_idx = self.categories_list.index(category)
        line_tensor = self.line_to_numpy(line)
        return category, line, category_idx, line_tensor

    def category_idx_to_tensor(self, category_indices):
        c = np.zeros((self.n_categories, len(category_indices)))
        for i, cat in enumerate(category_indices):
            c[cat, i] = 1
        return c
