#!/bin/bash
# Run this as a shell script to setup the environment

ENVNAME=interactenv

# conda create -n $ENVNAME python=3.9 -y
# source activate $ENVNAME
# conda install pytorch=*=*cuda11.8* torchvision torchaudio -c pytorch -c nvidia -y
# conda install -c conda-forge -c schrodinger pymol -y
# conda install pyg pytorch-cluster pytorch-scatter -c pyg -y
# conda install -c conda-forge rdkit -y
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install colorlog
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install wandb
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install e3nn
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install umap-learn
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install matplotlib
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install seaborn
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install plotly
# /n/home13/afang/.conda/envs/$ENVNAME/bin/pip install pynvml

# conda deactivate 



conda create -n $ENVNAME python=3.9 -y
source activate $ENVNAME
conda install numpy==1.26.4 -y
pip3 install torch==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install torch_scatter torch_cluster --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
pip3 install tensorboard
pip3 install e3nn
pip3 install wandb
pip3 install scipy
pip3 install rdkit-pypi
pip3 install openbabel-wheel
pip3 install biopython
pip3 install biotite

# plotting
pip3 install umap-learn
pip3 install matplotlib
pip3 install seaborn
pip3 install plotly