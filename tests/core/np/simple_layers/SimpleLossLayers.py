import numpy as np
from core import debug, info

class Softmax:

    def __init__(self):
        self.input = None
        self.outgoing_grad = None
        self.output = None

    def forward(self, x,  **kwargs):
        self.input = x
        numerator = np.exp(x)
        denominator = np.sum(numerator)
        self.output = numerator/denominator
        return self.output

    def backprop(self, incoming_gradient, **kwargs):
        n = self.input.shape[0]
        del_s = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i==j:
                    del_s[i,j] = self.output[i,0]-self.output[i,0]**2
                else:
                    del_s[i,j] = -self.output[i,0]*self.output[j,0]
        debug("[Softmax.backprop()] incoming_gradient = np.{}".format(repr(incoming_gradient)))
        debug("[Softmax.backprop()] del_s = np.{}".format(repr(del_s)))

        self.outgoing_grad = del_s*incoming_gradient

        return self.outgoing_grad


class NegativeLog:
    def __init__(self):
        self.input = None
        self.outgoing_grad = None
        self.output = None

    def forward(self, x,  **kwargs):
        self.input = x
        self.output = np.log(x)
        return self.output

    def backward(self, incoming_gradient, **kwargs ):
        self.outgoing_grad = incoming_gradient*1/self.input
        return self.outgoing_grad


class LogitType:
    def __init__(self):
        self.predicted = None
        self.target = None
        self.outgoing_grad = None
        self.output = None

    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        self.output = -np.sum(self.predicted*self.target)
        return self.output

    def backward(self, incoming_gradient, **kwargs):
        return self.target*incoming_gradient


class SoftmaxCrossEntropy:

    def __init__(self):
        self.probabilities, self.predicted, self.target = None, None, None
        self.logs, self.incoming_gradient, self.loss = None, None, None

    def forward(self, predicted, target):
        self.predicted = predicted
        max_values = np.max(self.predicted, axis=0)
        values = self.predicted - max_values
        #values = predicted
        numerator = np.exp(values)
        denominator = np.sum(numerator, axis=0)
        self.probabilities = numerator/denominator
        self.logs = np.log(self.probabilities)
        self.target = target
        self.loss = -np.sum(self.logs*target)
        return self.loss

    def backward(self, incoming_gradient):
        self.incoming_gradient = incoming_gradient
        return incoming_gradient*(self.probabilities - self.target)




