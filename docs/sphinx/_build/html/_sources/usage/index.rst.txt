API Description
===============

The API uses a network of nodes. Base class for all nodes is
MComputeNode. It contains forward and backward methods for evaluating the
network value in forward direction and propagating gradients back. Input
to the network is provided as numpy arrays set in dict objects
(:py:class:`core.np.Nodes.ComputeContext`) that the network passes along.

VarNode
--------
A VarNode  extracts its value  by reading it from the dict
object (:py:class:`core.np.Nodes.ComputeContext` object) passed through the
network in forward and backward evaluation steps. In the following code,
when you print the value first time, you will get None, printing after
the forward call will print the value contained in the key 'x' of the
ctx object.

.. code-block:: python

    import numpy as np
    import core.np.Nodes as node
    x_node = node.VarNode('x')
    x_value = np.array([[1,3,.9],[3,6,-.2]])
    ctx = node.ComputeContext({'x':x_value})
    print(x_node.value() )
    x_node.forward(ctx)
    print(x_node.value())


The output should be something like
::

  None
  [[ 1.   3.   0.9]
  [ 3.   6.  -0.2]]


Dense node (and other nodes)
----------------------------
The following code creates a network of three dense layers
connected by two RelU layers and Log loss.

.. code-block:: python

        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        linear1 = node.DenseLayer(x_node, 100, name="Dense-First")
        relu1 = act.RelUNode(linear1, name="RelU-First")
        linear2 = node.DenseLayer(relu1, 200, name="Dense-Second")
        relu2 = act.RelUNode(linear2, name="RelU-Second")
        linear3 = node.DenseLayer(relu2, 10, name="Dense-Third")
        loss_node = loss.LogitsCrossEntropy(linear3, yt_node, name="XEnt")


This network, we have two nodes that initiate the forward computation which
ends at the loss_node node. A forward pass will look something like

.. code-block:: python

        ctx = node.ComputeContext()
        x , y = get_from_train_set()
        ctx['x'], ctx['yt'] = x, to_one_hot(y, max_cat_num=9)
        x_node.forward(ctx)
        yt_node.forward(ctx)

At this point, the network will forward the computation all the way to the cross_entropy
node. You can get the value using ``loss_node.value()`` . In this case, you will get
a single number. To calculate  gradients for all the layers, use backward at the
layer from where you wish to initiate the computation.

.. code-block:: python

        incoming_grad = 1.0 # since we have a single number as loss.
        loss_node.backward(incoming_grad, self, var_map)


Finally, to update gradients, run the optimizer from last layer using an optimizer
function (not shown here)

.. code-block:: python

        loss_node.optimizer_step(optimizer_function, ctx)

Optimizer
---------
While the above discussion shows you how you can run the network forward and backward
and update the gradient, there is an easier way to do this - use an OptimizerIterator
along with a WeightUpdater as follows:

.. code-block:: python

        weight_updater = optim.AdamOptimizer()
        start_nodes = [x_node, y_node]
        optimizer = optim.OptimizerIterator(start_nodes, loss_node, weight_updater)
        loss = optimizer.step(ctx, 1.0) / batch_size






