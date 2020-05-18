from core.np.datasets.Mnist784Dataset import Mnist784
from core import debug, info, log_at_info
from tests.core.np.BaseTests import BaseComputeNodeTest
import numpy as np
import matplotlib.pyplot as plt
import core.np.Convolution as conv


class Mnist784DsTest(BaseComputeNodeTest):
    def test_load_one(self):
        mnist = Mnist784()
        for x, y in mnist.train_iterator(1):
            print(repr(y))
            x_img = x.reshape(28,28)/255
            plt.imshow(x_img,cmap='gray', vmin=0, vmax=255)
            plt.show()


        #input_node =
        #conv_node = conv.Convolution2D()


