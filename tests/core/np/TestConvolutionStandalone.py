import unittest

from core.np.Loss import L2DistanceSquaredNorm
from core import info, log_at_info
import numpy as np
import core.np.Nodes as node
import core.np.Convolution as conv
import matplotlib.pyplot as plt
from core import debug
import os
from . import BaseComputeNodeTest


class ConvolutionTests(BaseComputeNodeTest):

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def test_convolution_small(self):
        img = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 1, 3], [0, 2, -1, 4]])
        kernel = np.array([[1, -1], [0, 2]])
        img_node = node.VarNode('img')

        c2d = conv.Convolution2D(img_node, input_shape=(4, 4), kernel=kernel)
        var_map = {'img': img}
        img_node.forward(var_map, None, self)
        info("Original x into the convolution layer")
        info(repr(img))
        output_image = c2d.value(var_map)
        info("Output of the convolution layer")
        expected_output = np.array([[7., 9., 11.],
                                    [-1., 1., 5.],
                                    [3., -3., 6.]])
        np.testing.assert_array_almost_equal(expected_output, output_image)
        info(repr(output_image))
        log_at_info()
        c2d.backward(output_image * 0.1, self, var_map, "")
        info("Kernel before gradient descent")
        info(repr(c2d.get_kernel()))

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad

        optimizer = node.OptimizerIterator([img_node], c2d, optimizer_function)
        loss = optimizer.step(var_map, np.ones_like(output_image))
        info("Printing loss matrix - not really loss  but just the output of the last node")
        info(repr(loss))
        info("Printing kernel after gradient descent")
        info(repr(c2d.get_kernel()))
        expected_kernel = np.array([[0.998, -1.003111],
                                    [-1.444444444e-3, 1.9973333]])
        info("kernel gradient:{}".format(repr(c2d.kernel_grad)))
        np.testing.assert_array_almost_equal(expected_kernel, c2d.get_kernel())
        self.assertAlmostEqual(-0.001, c2d.get_bias())
        info("Bias after gradient descent:{}".format(c2d.get_bias()))
        info("Gradient of bias :{}".format(c2d.bias_grad))

    def test_convolution_with_l2(self):
        img = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 1, 3], [0, 2, -1, 4]])
        kernel = np.array([[1, -1], [0, 2]])
        y = np.ones((3, 3))

        img_node = node.VarNode('img')
        c2d = conv.Convolution2D(img_node, input_shape=(4, 4), kernel=kernel)
        target_node = node.VarNode('y')
        l2 = L2DistanceSquaredNorm(c2d, target_node)

        var_map = {'img': img, 'y': y}
        img_node.forward(var_map, None, self)
        target_node.forward(var_map, None, self)

        info("Original x into the convolution layer")
        info(repr(img))
        output_image = c2d.value(var_map)
        info("Output of the convolution layer")
        expected_output = np.array([[7., 9., 11.],
                                    [-1., 1., 5.],
                                    [3., -3., 6.]])
        np.testing.assert_array_almost_equal(expected_output, output_image)
        info(repr(output_image))
        log_at_info()
        info("Kernel before gradient descent")
        info(repr(c2d.get_kernel()))

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad

        optimizer = node.OptimizerIterator([img_node, target_node], l2, optimizer_function)
        loss = optimizer.step(var_map, 1.0)
        info("Took a single gradient descent step - calculated weights and updated gradients")
        info("<<<<Printing loss matrix after single step>>>>")
        info(repr(loss))
        info("Printing kernel:")
        info(repr(c2d.get_kernel()))
        info("--------------------------------------")
        info("Printing kernel gradient:")
        info(repr(c2d.get_kernel_grad()))
        info("-------------------------")
        info("Bias :{}".format(c2d.get_bias()))
        info("Bias gradient :{}".format(c2d.get_bias_grad()))
        expected_kernel = np.array([[0.98466667, -1.02288889],
                                    [-0.02066667, 1.96355556]])
        np.testing.assert_array_almost_equal(expected_kernel, c2d.get_kernel())
        expected_kernel_grad = np.array([[15.33333333, 22.88888889],
                                         [20.66666667, 36.44444444]])
        np.testing.assert_array_almost_equal(expected_kernel_grad, c2d.get_kernel_grad())
        expected_bias = -0.0064444444444444445
        expected_bias_grad = 6.444444444444445
        np.testing.assert_almost_equal(expected_bias, c2d.get_bias())
        np.testing.assert_almost_equal(expected_bias_grad, c2d.get_bias_grad())

    def get_image(self, filename):
        myfile = os.path.dirname(os.path.realpath(__file__))
        project_path = os.path.join(myfile, '..','..','..')
        imgpath = os.path.join(project_path, 'test.data','conv2d',filename)
        return  imgpath

    def test_convolution2d_plotting(self):
        image_path = self.get_image( 'Vd-Orig.png')
        image = plt.imread(image_path)
        shape = image.shape
        print("shape {}".format(shape))

        img_node = node.VarNode('x')
        x_image = self.rgb2gray(image * 20)
        print(x_image.shape)
        plt.imshow(x_image)
        plt.show()
        debug("Now showing ..")
        var_map = {'x': x_image}
        x_shape = (image.shape[0], image.shape[1])
        conv_node = conv.Convolution2D(img_node, x_shape)
        img_node.forward(var_map, None, self)
        final_image = conv_node.value(var_map)
        plt.imshow(final_image)
        plt.show()
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        img_node = node.VarNode('x')
        conv_node = conv.Convolution2D(img_node, x_shape, kernel=edge_kernel)
        img_node.forward(var_map, None, self)
        edge_img = conv_node.value(var_map)
        plt.imshow(edge_img)
        plt.show()

    def test_contribution_tracker(self):
        tracker = conv.ConvContributionTracker((3, 4))
        tracker.add_contribution(2, 1, (3, 4), (7, 4))
        tracker.add_contribution(2, 1, (3 + 1, 4 + 1), (7 - 1, 4 - 1))
        contribs = tracker.get_contributions(2, 1)
        self.assertEqual(len(contribs), 2)
        for dicts in contribs:
            print("W:{} , Y={}".format(dicts['w'], dicts['y']))


if __name__ == '__main__':
    unittest.main()
