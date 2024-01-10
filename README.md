Adapted from https://anonymous.4open.science/r/GET-anonymous/

# Implementation for Generalist Equivariant Transformer (GET)

## Requirements
We have prepare the configure for creating the environment with conda in *env.yml*:
```bash
conda env create -f env.yml
```

## Data Preparation
We assume the datasets are downloaded to the folder *./datasets*.

### 1. Data for Protein-Protein Affinity (PPA)

First download and decompress the protein-protein complexes in [PDBbind](http://www.pdbbind.org.cn/download.php) (registration is required):

```bash
wget http://www.pdbbind.org.cn/download/PDBbind_v2020_PP.tar.gz -P ./datasets/PPA
tar zxvf ./datasets/PPA/PDBbind_v2020_PP.tar.gz -C ./datasets/PPA
rm ./datasets/PPA/PDBbind_v2020_PP.tar.gz
```

Then process the dataset with the provided script:

```bash
python scripts/data_process/process_PDBbind_PP.py \
    --index_file ./datasets/PPA/PP/index/INDEX_general_PP.2020 \
    --pdb_dir ./datasets/PPA/PP \
    --out_dir ./datasets/PPA/processed
```

The processed data will be saved to *./datasets/PPA/processed/PDBbind.pkl*.

We still need to prepare the test set, i.e. Protein-Protein Affinity Benchmark Version 2. We have provided the index file in *./datasets/PPAB_V2.csv*, but the structure files need to be downloaded from the [official site](https://zlab.umassmed.edu/benchmark/). In case the official site is down, we also uploaded [a backup on Zenodo](https://zenodo.org/record/8318025/files/benchmark5.5.tgz?download=1).

```bash
wget https://zlab.umassmed.edu/benchmark/benchmark5.5.tgz -P ./datasets/PPA
tar zxvf ./datasets/PPA/benchmark5.5.tgz -C ./datasets/PPA
rm ./datasets/PPA/benchmark5.5.tgz
```

Then process the test set with the provided script:

```bash
python scripts/data_process/process_PPAB.py \
    --index_file ./datasets/PPA/PPAB_V2.csv \
    --pdb_dir ./datasets/PPA/benchmark5.5 \
    --out_dir ./datasets/PPA/processed
```

The processed dataset as well as different splits (Rigid/Medium/Flexible/All) will be saved to *./datasets/PPA/processed*.

### 2. Data for Ligand Binding Affinity (LBA)

You only need to download and decompress the LBA dataset:

```bash
mkdir ./datasets/LBA
wget "https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz?download=1" -O ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz
tar zxvf ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz -C ./datasets/LBA
rm ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz
```

### 3. Data for Ligand Efficacy Prediction (LEP)

You only need to download and decompress the LEP dataset:

```bash
mkdir ./datasets/LEP
wget "https://zenodo.org/record/4914734/files/LEP-split-by-protein.tar.gz?download=1" -O ./datasets/LEP/LEP-split-by-protein.tar.gz
tar zxvf ./datasets/LEP/LEP-split-by-protein.tar.gz -C ./datasets/LEP
rm ./datasets/LEP/LEP-split-by-protein.tar.gz
```


### 4. Data for PDBBind Benchmark

First download and extract the raw files:

Source of processed files: https://zenodo.org/records/8102783 Protein-ligand structures from PDBBind (v2019), and protein structures from the enzyme dataset as described in the paper: "Multi-Scale Representation Learning on Proteins" with associated code at https://github.com/vsomnath/holoprot.

```bash
mkdir ./datasets/PDBBind
wget "https://zenodo.org/record/8102783/files/pdbbind_raw.tar.gz?download=1" -O ./datasets/PDBBind/pdbbind_raw.tar.gz
tar zxvf ./datasets/PDBBind/pdbbind_raw.tar.gz -C ./datasets/PDBBind
rm ./datasets/PDBBind/pdbbind_raw.tar.gz
```

Then process the dataset with the provided script:
```bash
python scripts/data_process/process_PDBbind_benchmark.py \
    --benchmark_dir ./datasets/PDBBind/pdbbind \
    --out_dir ./datasets/PDBBind/processed
```

What is different here is that if you want to use fragment-based representation of small molecules, you need to process the data here:

```bash
python scripts/data_process/process_PDBbind_benchmark.py \
    --benchmark_dir ./datasets/PDBBind/pdbbind \
    --fragment PS_300 \
    --out_dir ./datasets/PDBBind/processed_PS_300
```

### 5. Data for Zero-Shot Inference on Nucleic-Acid-Ligand Affinity(NLA)

We need to use protein-protein data, protein-nucleic-acid data, and protein-ligand data for training, then evaluate the zero-shot performance on nucleic-acid-ligand affinity. All the data are extracted from PDBBind database. We have got protein-protein data in PPA and protein-ligand data in LBA, now we further need to get other data.
To get protein-nucleic-acid data:

```bash
wget http://www.pdbbind.org.cn/download/PDBbind_v2020_PN.tar.gz -P ./datasets/PN
tar zxvf ./datasets/PN/PDBbind_v2020_PN.tar.gz -C ./datasets
rm ./datasets/PN/PDBbind_v2020_PN.tar.gz
```
Then process the data:

```bash
python scripts/data_process/process_PDBbind_PN.py \
    --index_file ./datasets/PN/index/INDEX_general_PN.2020 \
    --pdb_dir ./datasets/PN \
    --out_dir ./datasets/PN/processed
```

To get nucleic-acid-ligand data:

```bash
wget http://www.pdbbind.org.cn/download/PDBbind_v2020_NL.tar.gz -P ./datasets/NL
tar zxvf ./datasets/NL/PDBbind_v2020_NL.tar.gz -C ./datasets
rm ./datasets/NL/PDBbind_v2020_NL.tar.gz
```

Then process the data:

```bash
python scripts/data_process/process_PDBbind_NL.py \
    --index_file ./datasets/NL/index/INDEX_general_NL.2020 \
    --pdb_dir ./datasets/NL \
    --out_dir ./datasets/NL/processed
```

## Training

### Ligand Binding Affinity (LBA)
These hyperparameters can be varied, they are not final
```bash
python train.py --gpu 0 --task PLA --lr 1e-3 --final_lr 1e-6 --max_epoch 200 --save_topk 3 --max_n_vertex_per_gpu 1000 --shuffle --model_type InteractNN --hidden_size 32 --n_layers 6\
 --radial_size 16 --n_channel 1 --n_rbf 16 --n_head 4 --k_neighbors 9 \
 --valid_set ./datasets/LBA/split-by-sequence-identity-30/data/val\
 --train_set ./datasets/LBA/split-by-sequence-identity-30/data/train\
 --save_dir ./datasets/LBA/split-by-sequence-identity-30/models/InteractNN\
 --seed 2023 --run_name testing --hierarchical --use_wandb
```

### Protein-Protein Affinity (PPA)
These hyperparameters can be varied, they are not final
```bash
python train.py --gpu 0 --task PPA --lr 0.0001 --final_lr 0.0001 --max_epoch 200 --save_topk 3 --max_n_vertex_per_gpu 1000 --shuffle --model_type InteractNN --hidden_size 32 --n_layers 6\
 --radial_size 16 --n_channel 1 --n_rbf 16 --n_head 4 --k_neighbors 9 \
 --valid_set ./datasets/PPA/processed/split0/valid.pkl\
 --train_set ./datasets/PPA/processed/split0/train.pkl\
 --save_dir ./datasets/PPA/processed/split0/models/InteractNN\
 --seed 2023 --run_name testing --hierarchical --use_wandb 
```

## Evaluation
### Ligand Binding Affinity (LBA)
Find saved location of the model and best epoch by looking at `$SAVE_DIR/version_$X/checkpoint/topk_map.txt`
```bash
SAVE_DIR="see training command"
VERSION="see save dir"
EPOCH="epochA_stepB"

python inference.py --test_set ./datasets/LBA/split-by-sequence-identity-30/data/test --task PLA \
 --ckpt $SAVE_DIR/version_$X/checkpoint/$EPOCH.ckpt\
 --save_path $SAVE_DIR/version_$X/results_$EPOCH.jsonl --batch_size 32 --num_workers 4 --gpu 0

python evaluate.py --task PLA --predictions $SAVE_DIR/version_$X/results_$EPOCH.jsonl
```

### Protein-Protein Affinity (PPA)
Find saved location of the model and best epoch by looking at `$SAVE_DIR/version_$X/checkpoint/topk_map.txt`
```bash
SAVE_DIR="see training command"
VERSION="see save dir"
EPOCH="epochA_stepB"

python inference.py --test_set ./datasets/PPA/processed/PPAB_V2.pkl --task PPA \
 --ckpt $SAVE_DIR/version_$X/checkpoint/$EPOCH.ckpt\
 --save_path $SAVE_DIR/version_$X/results_$EPOCH.jsonl --batch_size 32 --num_workers 4 --gpu 0

python evaluate.py --task PPA --predictions $SAVE_DIR/version_$X/results_$EPOCH.jsonl
```

## Experiments

### Protein-Protein Affinity (PPA)

We have provided the script for splitting, training and testing with 3 random seeds:

```bash
python scripts/exps/PPA_exps_3.py \
    --pdbbind ./datasets/PPA/processed/PDBbind.pkl \
    --ppab_dir ./datasets/PPA/processed \
    --config ./scripts/exps/configs/PPA/get.json \
    --gpus 0
```

### Ligand Binding Affinity (LBA)

We have provided the script for training and testing with 3 random seeds:

```bash
python scripts/exps/exps_3.py \
    --config ./scripts/exps/configs/LBA/get.json \
    --gpus 0
```

### Ligand Efficacy Prediction (LEP)

We have provided the script for training and testing with 3 random seeds:

```bash
python scripts/exps/exps_3.py \
    --config ./scripts/exps/configs/LEP/get.json \
    --gpus 0
```

### Universal Learning of PPA and LBA

To enhance the performance on PPA with additional data on LBA:

```bash
python scripts/exps/mix_exps_3.py \
    --config ./scripts/exps/configs/MIX/get_ppa.json \
    --gpus 0
```

To enhance the performance on LBA with additional data on PPA:

```bash
python scripts/exps/mix_exps_3.py \
    --config ./scripts/exps/configs/MIX/get_lba.json \
    --gpus 0
```

### PDBBind Benchmark

```bash
python scripts/exps/exps_3.py \
    --config ./scripts/exps/configs/PDBBind/identity30_get.json \
    --gpus 0
```

If you want to use fragment-based representation of small molecules, please replace the config with `identity30_get_ps300.json`.

### Zero-Shot on Nucleic Acid and Ligand Affinity

This experiment needs two 12G GPU (so a total of 2000 vertexes in a batch).

```bash
python scripts/exps/NL_zeroshot.py \
    --config scripts/exps/configs/NL/get.json \
    --gpu 1 2
```