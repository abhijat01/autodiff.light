Jupyer notebook with pytorch code
=================================

You will find several jupyter notebooks in the autodiff.light/jupyter directory, in particular
in the core.np folder. These notebooks are used mostly to compare pyunit test results with
pytorch results. Some code is also used as reference.

Pytorch reference code
-----------------------
 * **SingleStep_Pytoch_Grad_Example** contains code for calculating gradient across
   a single dense layer. The corresponding pyunit test case is in :py:meth:`tests.np.core.TestDenseLayer.DenseLayerStandAlone.test_single_step`