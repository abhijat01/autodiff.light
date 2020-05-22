# About 

Code for explaining fundamentals of back propagation on a compute graph. This 
uses computational primitives e.g., additions and multiplication, to build the graph 
for forward computation and back propagation. It results in  a slighlty more verbose 
code while setting up the network but everything also becomes very explicit, hence easy to understand. 

I am   

# Code 
## Main code 
Main code is in core and core.np packages. core.np package contains code that works with 
matrices, core package has code for simple functions. 

## tests
Tests are in "tests" directory. 

## Jupyter 
I have also committed some juputer notebooks that I am using to run equivalent pytorch 
computations to compare the results. These are in "jupyter" folder. 

# Prerequisites 

## Code 
For the code, you primarily need numpy but I have also used networkx and pyvis for 
visualization which I should probably remove

### Installing prerequisites 
#### pip 
    
I have manually changed the contents and removed most of the items listed when using 
pip freeze. Create a new conda or basic virtual environment, activate it  and then run the 
following command from within the environment 

    pip install -r pip-requirements.txt 
    
#### Conda 
Using conda does not yet work 

Using conda - this has been created using 

    conda list --export > conda-requirements.txt 
    
You can create a new conda environment using the command 

    conda create --name <envname> --file conda-requirements.txt
    
## Jupyter notebooks 

Jupyter notebooks require pytorch for most of the notebooks. There is one notebook that uses 
symy. You can skip this notebook if you do not wish to install sympy 

## Running pyunit on command prompt 

    python -m  unittest  tests.core.np.TestDenseLayer.DenseLayerStandAlone.test_linear_optimization


