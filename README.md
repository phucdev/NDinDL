# NDinDL
Implementation of Relational GCNs as part of the "New Developments in Deep Learning" seminar at HU Berlin

## Getting started
### Create and activate a conda environment
This also installs the project requirements (platform: win-64):

    conda create --name NDinDL --file requirements.txt python=3.8
    conda activate NDinDL

### Install the IPython kernel
In order to use the environment within jupyter notebook you have to install the IPython kernel.
Make sure to activate the conda environment first.

    python -m ipykernel install --user --name NDinDL --display-name "Python (NDinDL)"
Then when you open a jupyter notebook, you can select the kernel for the conda environment.

## Run the jupyter notebook
Now you can play around with our jupyter notebook. E.g.
 
    jupyter-lab NDinDL_R_GCNs.ipynb

## Troubleshooting
On Windows you may need to install pywin32 via conda to avoid ImportError when trying to launch jupyterlab.

    conda install pywin32