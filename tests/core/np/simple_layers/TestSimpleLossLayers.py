import unittest
import tests.core.np.simple_layers.SimpleLossLayers as sl
from core import  debug, info
import numpy as np 


class SoftmaxTestCase(unittest.TestCase):
    def test_forward(self):

        predicted = np.array([1, 3, -1., 0]).reshape(-1,1)
        target = np.array([0,1,0,0]).reshape(-1,1)

        softmax = sl.Softmax()
        output = softmax.forward(predicted)
        debug("[SoftmaxTestCase.test_forward()] output = np.{}".format(repr(output)))
        neg_log = sl.NegativeLog()
        output = neg_log.forward(output)
        debug("[SoftmaxTestCase.test_forward()] After negative log output = np.{}".format(repr(output)))
        logit = sl.LogitType()
        output = logit.forward(output, target)
        debug("[SoftmaxTestCase.test_forward()] After logit, output = np.{}".format(repr(output)))
        grad = logit.backward(1.0) 
        debug("[SoftmaxTestCase.test_forward()] Grad from logit = np.{}".format(repr(grad)))
        grad = neg_log.backward(grad)
        debug("[SoftmaxTestCase.test_forward()] grad after negative log= np.{}".format(repr(grad)))
        grad = softmax.backprop(grad)
        debug("[SoftmaxTestCase.test_forward()] grad after softmax= np.{}".format(repr(grad)))

        sx = sl.SoftmaxCrossEntropy()
        loss = sx.forward(predicted, target)
        debug("[SoftmaxTestCase.test_forward()] loss from Softmax XEntropy= {}".format(repr(loss)))
        grad = sx.backward(1.0)
        debug("[SoftmaxTestCase.test_forward()] grad  SoftmaxCrossEntropy= np.{}".format(repr(grad)))

        debug("-------------  Batch with 2 samples -----------------")
        predicted = np.array([[1, 3, -1, 0], [0, 9, 1, 3.]]).reshape(-1,2)
        target = np.array([[0,1,0,0],[0,0,0,1]]).reshape(-1,2) 
        loss = sx.forward(predicted, target) 
        debug("[SoftmaxTestCase.test_forward()] Loss = np.{}".format(repr(loss)))
        grad = sx.backward(1.0)
        debug("[SoftmaxTestCase.test_forward()] grad = np.{}".format(repr(grad)))


    def test_batch(self):
        sx = sl.SoftmaxCrossEntropy()
        debug("-------------  Batch with 2 samples -----------------")
        predicted = np.array([[1, 3, -1, 0], [0, 9, 1, 3.]]).T.reshape(4,2)
        debug("[SoftmaxTestCase.test_batch()] predicted = np.{}".format(repr(predicted)))
        target = np.array([[0,1,0,0],[0,0,0,1]]).T.reshape(4,2)
        loss = sx.forward(predicted, target)
        debug("[SoftmaxTestCase.test_forward()] Loss = {}".format(repr(loss)))
        grad = sx.backward(1.0)
        debug("[SoftmaxTestCase.test_forward()] grad = np.{}".format(repr(grad.T)))


if __name__ == '__main__':
    unittest.main()
