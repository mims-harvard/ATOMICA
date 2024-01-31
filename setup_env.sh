#!/bin/bash
# Run this as a shell script to setup the environment

ENVNAME=interactenv

conda create -n $ENVNAME python=3.9 -y
source activate $ENVNAME
conda install pytorch=*=*cuda11.8* torchvision torchaudio -c pytorch -c nvidia -y
conda install -c conda-forge -c schrodinger pymol -y
conda install pyg pytorch-cluster pytorch-scatter -c pyg -y
conda install -c conda-forge rdkit -y
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install colorlog
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install wandb
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install e3nn
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install umap-learn
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install matplotlib
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install seaborn
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install plotly
/n/home13/afang/.conda/envs/$ENVNAME/bin/pip install pynvml

conda deactivate 
