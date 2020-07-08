import unittest

import core.np.Nodes as n
import core.np.rnn.Recurrent as rnn
import core.np.Loss as loss
import numpy as np
from core import log_at_info, debug, info
import core.np.Optimization as autodiff_optim
from tests.core.np.BaseTests import BaseComputeNodeTest
import os
from tests.core.np.rnn.NamesDs import NameDS
import time


class RnnLayerFullTests(BaseComputeNodeTest):

    def setUp(self):
        data_dir = os.path.expanduser("~")
        self.data_dir = os.path.join(data_dir, "ai-ws", "rnn-basics", "data", "names")
        self.name_ds = NameDS(self.data_dir)

    def test_train(self):
        num_iter = 100000
        x_node = n.VarNode('x')
        y_target_node = n.VarNode('y_target')
        rnn_node = rnn.SimpleRnnLayer(x_node, self.name_ds.n_categories, 15)
        loss_node = loss.SoftmaxCrossEntropy(rnn_node, y_target_node)
        all_losses = []
        optimizer_func = autodiff_optim.AdamOptimizer()
        optimizer_func = autodiff_optim.SGDOptimizer(lr=0.0001)
        optimizer = autodiff_optim.OptimizerIterator([x_node, y_target_node], loss_node, optimizer_func)
        ctx = n.ComputeContext({'x': "", 'y_target': ""})
        log_at_info()
        every = 500
        t = time.time()
        for i in range(1, num_iter+1):
            rnn_node.set_initial_state_to_zero()
            c, l, category_index, name_tensor = self.name_ds.random_training_example()
            cat_tensor = self.name_ds.category_idx_to_tensor([category_index])
            ctx['x'] = name_tensor
            ctx['y_target']=cat_tensor
            ctx['i'] = i
            loss_value = optimizer.step(ctx, 1.0)
            all_losses.append(loss_value)
            if i%every ==0 :
                t = time.time()-t
                last_10 = all_losses[-every:]
                av = np.average(last_10)
                info("[{:06d}] Avg. loss = {:10.6f}"
                     " | {:04.2f}s per {}  | Total Iters set to:{}".format( i , av, t, every, num_iter))
                all_losses=[]
                t = time.time()





