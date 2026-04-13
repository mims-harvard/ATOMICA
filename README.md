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

ATOMICA requires PyTorch with CUDA support. Please refer to the installation instructions in [setup](https://github.com/mims-harvard/ATOMICA/tree/main/setup) which provides instructions for setting up with uv or mamba/conda.


## :zap: Quick Start

Generate embeddings from list of PDB files with ATOMICA model in just a few lines. See the tutorial at [tutorials/1_get_embeddings](https://github.com/mims-harvard/ATOMICA/tree/main/tutorials/1_get_embeddings) for more details.

## :star: Other setup
Optional steps, only required if you plan on training your own ATOMICA model.

### Download All Processed Datasets
The data for pretraining and downstream analyses is hosted at [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX).

We provide the following datasets:
* Processed CSD and QBioLiP (based on PDB) interaction complex graphs for pretraining
* Processed datasets for four RNAGlib benchmark tasks: RNA-GO, RNA-Ligand, RNA-Protein, RNA-Site
* Processed datasets for the MASIF-Ligand benchmark.
* Processed datasets for the PPI and orthosteric inhibitors analysis.
* Processed protein interfaces of dark proteome binding sites to ion and small molecules

### Download All Model Checkpoints
Model checkpoints are provided on [Hugging Face](https://huggingface.co/ada-f/ATOMICA). The following models are available:
* ATOMICA pretrained model
* Finetuned ATOMICA-Ligand prediction models for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA
* Finetuned MaSIF-ligand pocket classification models (5 seeds) — protein pocket classification across 7 small-molecule ligands (ADP, CoA, FAD, heme, NAD, NAP, SAM)
* Finetuned RNAglib prediction models (5 seeds each) for four RNA structure-function tasks:
    * `rna_go` — RNA Gene Ontology term prediction (multi-label)
    * `rna_ligand` — RNA pocket ligand classification (multi-class)
    * `rna_protein` — RNA residue protein-binding prediction (binary)
    * `rna_site` — RNA residue small-molecule-binding prediction (binary)

### Training / Finetuning your own ATOMICA model
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

## :seedling: Tutorials
### Get embeddings from ATOMICA model
Refer to the tutorial at `tutorials/1_get_embeddings` for more details.

### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `tutorials/2_atomica_ligand` for an example of how to use the model for dark proteome ligand predictions.

### RNA structure-function prediction (RNAglib benchmarks)
Refer to `tutorials/3_rna_structure_function` for reproducing the ATOMICA paper results on four RNAglib benchmarks (RNA-GO, RNA-Ligand, RNA-Protein, RNA-Site) using the finetuned checkpoints.

### MaSIF-Ligand benchmark
Refer to `tutorials/4_atomica_masif_benchmark` for the protein pocket classification benchmark across 7 small-molecule ligands, using the finetuned checkpoints.

### PPI and orthosteric inhibitors
Refer to `tutorials/5_ppi_and_inhibitors` for comparing ATOMICA embeddings of orthosteric PPI inhibitors against embeddings of the native protein-protein / protein-peptide complexes they inhibit (2P2IDB).

### InteractScore: per-residue importance at an interface
Refer to the jupyter notebook at `tutorials/6_interact_score` for computing per-residue InteractScores at a protein-ligand interface via masked-embedding cosine similarity.

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