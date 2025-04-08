![ATOMICA logo](assets/atomica_logo.png)
# Learning Universal Representations of Intermolecular Interactions

**Authors**
* Ada Fang
* Zaixi Zhang
* Andrew Zhou
* Marinka Zitnik

[Paper](link_to_paper) | [Project Website](https://zitniklab.hms.harvard.edu/projects/ATOMICA)

ATOMICA is a geometric AI model that learns universal representations of molecular interactions at an atomic scale. The model is pretrained on 2,037,972 molecular interaction interfaces from the Protein Data Bank and Cambridge Structural Database, this includes protein-small molecule, protein-ion, small molecule-small molecule, protein-protein, protein-peptide, protein-RNA, protein-DNA, and nucleic acid-small molecule complexes. Embeddings of ATOMICA can be generated with the open source model weights and code to be used for various downstream tasks. In the paper, we demonstrate the utility of ATOMICA embeddings for studying the human interfaceome network with ATOMICANets and for annotating ions and small molecules to proteins in the dark proteome.

## :rocket: Installation and Setup

### 1. Download the Repository
Clone the Gihub Repository:
```bash
git clone https://github.com/mims-harvard/ATOMICA
cd ATOMICA
```

### 2. Set Up Environment
Set up the environment according to `setup_env.sh`.

### 3. (optional) Download Processed Datasets
The data for pretraining and downstream analyses is hosted at [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX).

We provide the following datasets:
* Processed CSD and QBioLiP (based on PDB) interaction complex graphs for pretraining
* Processed protein interfaces of human proteome binding sites to ion, small molecule, lipid, nucleic acid, and protein modalities
* Processed protein interfaces of dark proteome binding sites to ion and small molecules

### 4. Download Model Checkpoints
Model checkpoints are provided on [Hugging Face](https://huggingface.co/ada-f/ATOMICA). The following models are available:
* ATOMICA model
* Pretrained ATOMICA-Interface model
* Finetuned ATOMICA-Ligand prediction models for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA

## :star: Usage
### Train ATOMICA
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `case_studies/atomica_ligand/example_run_atomica_ligand.ipynb` for an example of how to use the model for binder prediction.

### Explore ATOMICANets
Refer to the jupyter notebook at `case_studies/atomica_net/example_atomica_net.ipynb`

### Embedding your own structures
Make sure to download the ATOMICA model weights and config files from [Hugging Face](https://huggingface.co/ada-f/ATOMICA).

**For embedding biomolecular complexes:** process .pdb files with `data/process_pdbs.py` and embed with `get_embeddings.py`.

**For embedding protein-(ion/small molecule/lipid/nucleic acid/protein) interfaces:** first predict (ion/small molecule/lipid/nucleic acid/protein) binding sites with [PeSTo](https://github.com/LBM-EPFL/PeSTo), second process the PeSTo output .pdb files with `data/process_PeSTo_results.py`, finally embed with `get_embeddings.py`.

## :bulb: Questions
For questions, please leave a GitHub issue or contact Ada Fang at <ada_fang@g.harvard.edu>.
