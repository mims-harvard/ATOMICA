![ATOMICA logo](assets/atomica_logo.png)
# Learning Universal Representations of Intermolecular Interactions

**Authors**
* Ada Fang
* Michael Desgagné
* Zaixi Zhang
* Andrew Zhou
* Joseph Loscalzo
* Bradley L. Pentelute
* Marinka Zitnik

[Preprint](https://www.biorxiv.org/content/10.1101/2025.04.02.646906) | [Project Website](https://zitniklab.hms.harvard.edu/projects/ATOMICA)

ATOMICA is a geometric AI model that learns universal representations of molecular interactions at an atomic scale. The model is pretrained on 2,037,972 molecular interaction interfaces from the Protein Data Bank and Cambridge Structural Database, this includes protein-small molecule, protein-ion, small molecule-small molecule, protein-protein, protein-peptide, protein-RNA, protein-DNA, and nucleic acid-small molecule complexes. Embeddings of ATOMICA can be generated with the open source model weights and code to be used for various downstream tasks. In the paper, we demonstrate the utility of ATOMICA embeddings for studying the human interfaceome network with ATOMICANets and for annotating ions and small molecules to proteins in the dark proteome.

## :rocket: Installation and Setup

### Installation (GPU with CUDA 11.8 - Recommended)

ATOMICA requires PyTorch with CUDA support. Follow these steps in order:

```bash
# Step 1: Install PyTorch with CUDA 11.8
pip install torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Step 2: Install PyTorch Geometric dependencies
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html

# Step 3: Install ATOMICA
pip install git+https://github.com/mims-harvard/ATOMICA.git
```

After installation, you can use the command-line tools:
```bash
atomica-embeddings --help
atomica-train --help
```

### Development Installation

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/mims-harvard/ATOMICA
cd ATOMICA

# Install PyTorch and dependencies first (see steps 1-2 above)
pip install torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html

# Install ATOMICA in editable mode
pip install -e .
```

### CPU-Only Installation (Not Recommended)

For CPU-only installation (much slower, not recommended for production use):
```bash
pip install torch==2.1.1
pip install git+https://github.com/mims-harvard/ATOMICA.git
```

**Note:** torch-scatter and torch-cluster may fail to build without CUDA. Use the GPU installation method for best results.

## :zap: Quick Start

Generate embeddings from a PDB file in just a few lines:

```python
from atomica.get_embeddings import main
import argparse

# Set up arguments
args = argparse.Namespace(
    model_ckpt="path/to/atomica_checkpoint.ckpt",  # Download from HuggingFace
    data_path="path/to/your_data.pkl",  # Processed PDB data
    output_path="embeddings.pkl",
    batch_size=4
)

# Generate embeddings
main(args)
```

Or use the command-line interface:
```bash
atomica-embeddings \
    --model_ckpt path/to/atomica_checkpoint.ckpt \
    --data_path path/to/your_data.pkl \
    --output_path embeddings.pkl \
    --batch_size 4
```

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

## :star: Training
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

## :seedling: Tutorials
### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `tutorials/atomica_ligand/example_run_atomica_ligand.ipynb` for an example of how to use the model for binder prediction.

### Explore ATOMICANets
Refer to the jupyter notebook at `tutorials/atomica_net/example_atomica_net.ipynb`

### Embedding your own structures
Make sure to download the ATOMICA model weights and config files from [Hugging Face](https://huggingface.co/ada-f/ATOMICA).

**For embedding biomolecular complexes:** process .pdb files with `data/process_pdbs.py` and embed with `get_embeddings.py`. See the tutorial for data processing at `data/README.md` [here](https://github.com/mims-harvard/ATOMICA/tree/main/data) and the examples at `data/example`.

**For embedding protein-(ion/small molecule/lipid/nucleic acid/protein) interfaces:** first predict (ion/small molecule/lipid/nucleic acid/protein) binding sites with [PeSTo](https://github.com/LBM-EPFL/PeSTo), second process the PeSTo output .pdb files with `data/process_PeSTo_results.py`, finally embed with `get_embeddings.py`.

## :bulb: Questions
For questions, please leave a GitHub issue or contact Ada Fang at <ada_fang@g.harvard.edu>.

## :balance_scale: License
The code in this package is licensed under the MIT License.

## :scroll: Citation
If you use ATOMICA in your research, please cite the following [preprint](https://www.biorxiv.org/content/10.1101/2025.04.02.646906v1):
```
@article{fang2025atomica,
  title={Learning Universal Representations of Intermolecular Interactions with ATOMICA},
  author={Fang, Ada and Desgagné, Michael and Zhang, Zaixi and Zhou, Andrew and Loscalzo, Joseph, and Pentelute, Bradley L and Zitnik, Marinka},
  journal={In Review},
  url={https://www.biorxiv.org/content/10.1101/2025.04.02.646906},
  year={2025}
}
```