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
Model checkpoints are provided on [Hugging Face](https://huggingface.co/ada-f/ATOMICA/). The following models are available:
* Pretrained complex model
* Pretrained ATOMICA-Interface model
* Finetuned ATOMICA-Ligand prediction models for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA

## Usage
### Train ATOMICA
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

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
