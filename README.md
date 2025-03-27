# ATOMICA: Universal Geometric AI for Molecular Interactions across Biomolecular Modalities

**Authors**
* Ada Fang
* Zaixi Zhang
* Andrew Zhou
* Marinka Zitnik

## Overview of ATOMICA
TO DO: adapt from paper

## Installation and Setup

### 1. Download the Repository
First, clone the Gihub Repository:
```bash
git clone https://github.com/mims-harvard/ATOMICA
cd ATOMICA
```

### 2. Set Up Environment
Set up the environment according to `setup_env.sh`.

### 3. Download Processed Datasets
The data is hosted at <insert-link>. Please unzip once downloaded.

We provide the following datasets:
* Processed CSD and QBioLiP (based on PDB) interface complex graphs for pretraining
* Processed protein interfaces of human proteome binding sites to ion, small molecule, lipid, nucleic acid, and protein modalities
* Processed protein interfaces of dark proteome binding sites to ion and small molecules

To process other structures for embedding complexes refer to `data/process_pdbs.py`. To process other structures for embedding protein interfaces refer to **TO DO: PeSTo processing**. These can be used to generate embeddings, or finetuned for other tasks.

### 4. Download Model Checkpoints
Model checkpoints are provided for
* Pretrained complex model <insert-link>
* Pretrained protein interface model <insert-link>
* Finetuned protein binder prediction models <insert-link> for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA

## Usage
### Train ATOMICA
The model was trained on 4 NVIDIA H100 gpus in parallel with the following command:
```bash
~/.conda/envs/interactenv/bin/torchrun --nnodes=1 --nproc_per_node=4 --standalone train.py \
    --train_set datasets/QBioLiP_train.pkl \
    --valid_set datasets/QBioLiP_valid.pkl \
    --train_set2 datasets/CSD_PS_300_train.pkl \
    --valid_set2 datasets/CSD_PS_300_valid.pkl \
    --task pretrain_torsion_masking \
    --gpu 0 1 2 3 \
    --num_workers 32 \
    --lr 1e-4 \
    --final_lr 1e-6 \
    --max_epoch 200 \
    --shuffle \
    --atom_hidden_size 32 \
    --block_hidden_size 32 \
    --n_layers 4 \
    --edge_size 32 \
    --k_neighbors 8 \
    --max_n_vertex_per_gpu 1000 \
    --max_n_vertex_per_item 256 \
    --fragmentation_method PS_300 \
    --global_message_passing \
    --cycle_steps 400000 \
    --mask_proportion 0.1 \
    --torsion_noise 0.5 \
    --translation_noise 1 \
    --rotation_noise 0.25 \
    --max_rotation 0.5 \
    --rot_weight 0.1 \
    --save_dir model_checkpoints \
    --run_name ATOMICA
```

### Train ATOMICA-Interface
This model was finetuned from pretrained ATOMICA on protein interfaces with the following command:
```bash

~/.conda/envs/interactenv/bin/torchrun --nnodes=1 --nproc_per_node=4 --standalone train.py \
    --train_set datasets/QBioLiP_train.pkl \
    --valid_set datasets/QBioLiP_valid.pkl \
    --task prot_interface \
    --gpu 0 1 2 3 \
    --num_workers 32 \
    --lr 5e-4 \
    --max_epoch 50 \
    --shuffle \
    --atom_hidden_size 32 \
    --block_hidden_size 32 \
    --n_layers 4 \
    --edge_size 32 \
    --k_neighbors 8 \
    --max_n_vertex_per_gpu 1000 \
    --max_n_vertex_per_item 256 \
    --fragmentation_method PS_300 \
    --global_message_passing \
    --save_dir model_checkpoints \
    --pretrain_ckpt path/to/pretrained/ATOMICA/checkpoint \
    --run_name ATOMICA-Interface
```

### Train ATOMICA-Ligand
This model was finetuned from pretrained ATOMICA on protein binder pockets for a given binder with the following command:
```bash

python train.py \
    --train_set datasets/binder_train.pkl \
    --valid_set datasets/binder_valid.pkl \
    --task binary_classifier \
    --gpu 0 \
    --num_workers 8 \
    --lr 1e-4 \
    --max_epoch 50 \
    --shuffle \
    --atom_hidden_size 32 \
    --block_hidden_size 32 \
    --n_layers 4 \
    --edge_size 32 \
    --k_neighbors 8 \
    --max_n_vertex_per_gpu 1000 \
    --max_n_vertex_per_item 256 \
    --fragmentation_method PS_300 \
    --global_message_passing \
    --save_dir model_checkpoints \
    --pretrain_ckpt path/to/pretrained/ATOMICA/checkpoint \
    --run_name ATOMICA-Interface
```

### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `case_studies/binder_prediction/ATOMICA_Binder_Prediction.ipynb` for an example of how to use the model for binder prediction. **TODO: add jupyter notebook**

### Explore ATOMICANets
Refer to the jupyter notebook at `case_studies/human_interfaceome_network/ATOMICA_Network.ipynb`

## Additional Resources
* [ATOMICA Paper](link_to_paper)
* [ATOMICA Website](link_to_website)
* [Demo](link_to_demo)

## Questions
For questions, please leave a GitHub issue or contact Ada Fang at <ada_fang@g.harvard.edu>.
