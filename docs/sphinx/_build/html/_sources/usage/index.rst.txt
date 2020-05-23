API Description
===============

The API uses a network of nodes. The base class for all nodes is
MComputeNode. It contains forward and backward methods for evaluating the
network value in forward direction and propagating gradients back. Input
to the network is provided as numpy arrays set in dict objects that
the network passes along.

VarNode
--------
A VarNode is a node that makes it value available by reading it from the dict
object passed through the network in forward and backward evaluation steps.


Optimizer
---------
