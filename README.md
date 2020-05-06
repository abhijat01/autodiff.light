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


