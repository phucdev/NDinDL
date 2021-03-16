# NDinDL
Implementation of Relational GCNs as part of the "New Developments in Deep Learning" seminar at HU Berlin

## Getting started
### Create and activate a conda environment
    conda create --name NDinDL python=3.7
    conda activate NDinDL

### Prerequisites
Install [PyTorch](https://pytorch.org/get-started).
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Install project requirements
    conda install jupyterlab ipykernel

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