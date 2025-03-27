#!/bin/bash

ENVNAME=atomicaenv
conda create -n $ENVNAME python=3.9 -y
source activate $ENVNAME
conda install numpy==1.26.4 -y
pip3 install torch==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install torch_scatter torch_cluster --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
pip3 install tensorboard==2.18.0
pip3 install e3nn==0.5.1 # possibly not compatible with e3nn > 0.5.4
pip3 install scipy==1.13.1
pip3 install rdkit-pypi==2022.9.5
pip3 install openbabel-wheel==3.1.1.20
pip3 install biopython==1.84
pip3 install biotite==0.40.0
pip3 install atom3d
pip3 install wandb==0.18.2
pip3 install orjson

# plotting
pip3 install umap-learn
pip3 install matplotlib
pip3 install seaborn
pip3 install plotly