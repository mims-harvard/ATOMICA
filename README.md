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

ATOMICA runs on any CUDA build of PyTorch from **11.8 through 13.0**, and on CPU. Install the PyTorch wheel matching your NVIDIA driver, then install ATOMICA on top of it:

```bash
# 1. PyTorch for your CUDA version -- this is the only line that changes.
#    cu130 (CUDA 13.0) | cu128 (CUDA 12.8) | cu118 (CUDA 11.8) | cpu
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. ATOMICA
git clone https://github.com/mims-harvard/ATOMICA.git && cd ATOMICA
pip install -e ".[dev]"
```

No CUDA toolkit is needed on the host — the PyTorch wheel bundles its own CUDA runtime — and `torch-scatter`/`torch-cluster` are not required. See [setup/README.md](setup/README.md) for the driver-to-CUDA table, scripted installs with uv or mamba/conda, a Docker/Apptainer image, the table of tested configurations, and troubleshooting.


## :zap: Quick Start

Turn a list of PDB files into ATOMICA representations in two commands:

```bash
python -m atomica.data.process_pdbs \
  --data_index_file data/example/example_inputs.csv \
  --out_path data/example/example_processed_data.parquet

python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_z_graph.parquet \
  --representations z_graph --pool mean_std_global
```

ATOMICA produces eight named representations, at the atom, block, interface and graph levels, and
which one you want depends on what you will do with it. `python -m atomica.representations --guidance`
prints the choices. See the tutorial at [tutorials/1_get_embeddings](https://github.com/mims-harvard/ATOMICA/tree/main/tutorials/1_get_embeddings) for the full walkthrough.

## :star: Other setup
Optional steps, only required if you plan on training your own ATOMICA model.

### Download All Processed Datasets
The data for pretraining and downstream analyses is hosted at [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX).

We provide the following datasets:
* Processed CSD and QBioLiP (based on PDB) interaction complex graphs for pretraining
* Processed datasets for four RNAGlib benchmark tasks: RNA-GO, RNA-Ligand, RNA-Protein, RNA-Site
* Processed datasets for the MASIF-Ligand benchmark.
* Processed datasets for the ATP vs ADP nucleotide state benchmark.
* Processed datasets for the same-ligand pocket retrieval benchmark.
* Processed datasets for the PPI and orthosteric inhibitors analysis.
* Processed protein interfaces of dark proteome binding sites to ion and small molecules

### Download All Model Checkpoints
Model checkpoints are provided on [Hugging Face](https://huggingface.co/ada-f/ATOMICA). The following models are available:
* ATOMICA pretrained model
* Finetuned ATOMICA-Ligand prediction models for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA
* MaSIF-ligand: MaSIF-similar protein–ligand-excluded pretraining checkpoint
* RNAGlib:
    * Protein–RNA-excluded pretraining checkpoint
    * Nucleic-acid–ligand-excluded pretraining checkpoint
    * Finetuned RNA-GO prediction model
For the benchmarks, task-specific prediction heads are trained on frozen representations from the corresponding pretraining checkpoints which exclude overlapping data from pretraining.

### Training / Finetuning your own ATOMICA model
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

## :seedling: Tutorials
### Get representations from the ATOMICA model
Refer to `tutorials/1_get_embeddings` for the named representations, which one to use for which kind of task, and the commands that extract them.

### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `tutorials/2_atomica_ligand` for an example of how to use the model for dark proteome ligand predictions.

### RNA structure-function prediction (RNAglib benchmarks)
Refer to `tutorials/3_rna_structure_function` for reproducing the ATOMICA paper results on four RNAglib benchmarks (RNA-GO, RNA-Ligand, RNA-Protein, RNA-Site).

### MaSIF-Ligand benchmark
Refer to `tutorials/4_atomica_masif_benchmark` for the protein pocket classification benchmark across 7 small-molecule ligands.

### PPI and orthosteric inhibitors
Refer to `tutorials/5_ppi_and_inhibitors` for comparing ATOMICA embeddings of orthosteric PPI inhibitors against embeddings of the native protein-protein / protein-peptide complexes they inhibit (2P2IDB).

### ATOMICAScore: per-residue importance at an interface
Refer to the jupyter notebook at `tutorials/6_interact_score` for ranking the amino-acid residues at a protein-ligand interface by how much masking each one moves the model's representation of the ligand.

### ATP versus ADP nucleotide state
Refer to `tutorials/7_atp_adp_nucleotide_state` for classifying which nucleotide a binding site held, from the empty pocket alone.

### Same-ligand pocket retrieval
Refer to `tutorials/8_pocket_retrieval` for retrieving pockets that bind the same ligand across structurally distinct proteins, by cosine similarity between frozen embeddings.

### Metal coordination probes
Refer to `tutorials/9_metal_coordination` for linear probes that read a metal site's coordination number and coordination geometry off the frozen block representation.

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