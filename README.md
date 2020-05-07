# About 

Code for explaining fundamentals of back propagation on a compute graph. The idea is 
to use  computational primitives e.g., additions and multiplication, to build the graph 
and then use it for back propagation. It results in rather verbose code while setting up 
the network but everything also becomes very explicit, hence easy to understand. 

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
Using conda - this has been created using 

    conda list --export > conda-requirements.txt 
    
You can create a new conda environment using the command 

    conda create --name <envname> --file conda-requirements.txt
    
Using pip - this has been created using 
   
    pip freeze >  pip-requirements.txt 
    
You should create a new conda or basic virtual environment and then run the 
following command:

    pip install -r pip-requirements.txt 
    
## Jupyter notebooks 

Jupyter notebooks require pytorch. Any recent version should work. 

## Running pyunit on command prompt 

    python -m  unittest  tests.core.np.TestDenseLayer.DenseLayerStandAlone.test_linear_optimization


