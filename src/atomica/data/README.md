# Processing data for ATOMICA
To process your own structures to embed with ATOMICA please use the `data/process_pdbs.py` script.

## Biomolecule structure file formats supported
The script supports the following formats:
* `.pdb`, legacy PDB format
* `.cif`, mmCIF format

## Inputs
To run `data/process_pdbs.py` you need to provide the following inputs:

**Data Index File**
`--data_index_file`: this is a `.csv` file used to specify the interface of which pairs of chains or ligands you would like to process.
The file should contain the following columns:
* `pdb_id`: the PDB ID of the structure, or unique identifier of the structure.
* `pdb_path`: the path to the structure file, either `.pdb` or `.cif`.
* `chain1`: the chain ID of the first chain(s) in the first interface. If you would like to specify multiple chains for the first interface you can separate the chain IDs with an underscore `_`, e.g. `A_B_C`.
* `chain2`
    * For protein-protein, protein-peptide, protein-nucleic acid: this is the chain ID of the second chain(s) in the second interface. If you would like to specify multiple chains for the second interface you can separate the chain IDs with an underscore `_`, e.g. `A_B_C`.
    * For protein-ion, protein-small molecule, nucleic acid-ligand: this is the chain ID of the ligand for the second interface.
* `lig_code`: the CCD ligand code of the ligand in the interface. Leave this blank if the ligand is not a small molecule or ion.
* `lig_smiles`: the SMILES string of the ligand in the interface. Leave this blank if the ligand is not a small molecule.
* `lig_resi`: the integer residue index of the ligand in the interface. Leave this blank if the ligand is not a small molecule or ion.
* `label`: (optional) if you would like to finetune the model on your own labeled dataset, you can provide the label of the interface here.

If there are particular sections of a chain that you would not like in the interaction interface that are within the distance threshold, you should remove them from the PDB file before running the script.

**Other arguments**
* `--out_path`: the path to the output pickle file where the processed data will be saved.
* `--interface_dist_th`: the distance threshold for the interface. Residues who has atoms with distance below this threshold are considered in the complex interface. Default is 8.0 A.
* `--fragmentation_method`: the method used to fragment the small molecule structures into common chemical motifs for the block-level of the graph. It is recommended to turn this on by setting this to be `PS_300`.

**Example**
```
python -m atomica.data.process_pdbs \
    --data_index_file data/example/example_inputs.csv \
    --out_path data/example/example_outputs.parquet \
    --interface_dist_th 8.0 \
    --fragmentation_method PS_300
```

In the example data index file `data/example/example_inputs.csv`, we provide examples for the following PDB ids: 6llw (protein-small molecule), 3i5x (protein-RNA), 5kl2 (protein-DNA), 6d1u (protein-peptide), 2uxq (protein-protein), 6hrg (protein-ion), and 4yaz (nucleic acid-small molecule).

## Output format
The output is a parquet file containing the processed interaction complexes as a list of nested dictionaries. Each entry is a row in the parquet file and contains the following fields:
* `id`: the PDB ID of the structure, or unique identifier of the structure.
* `block_to_pdb_indexes`: a dictionary that maps from the `block_idx` to the indexes of the residues (chain, residue index) in the PDB file.
* `X`: list of floats of shape [Natom,3] which contains the 3D atomic coordinates of every atom as well as well as two special global atoms defined to be the center of their respective interfaces.
* `A`: list of integer atom element indexes of shape [Natom].
* `B`: list of integer block identity indexes of shape [Nblock].
* `block_lengths`: list of integers of shape [Nblock] which contains the number of atoms in each block.
* `segment_ids`: list of integers of shape [Nblock] which contains the segment id of each block. Segment id is 0 for the first interface and 1 for the second interface.
* `atom_positions`: deprecated.

## Embedding your own structures
To embed your own structures with ATOMICA, use `python -m atomica.representations`. It takes a processed data file and writes the representations you name for each interface. Run `python -m atomica.representations --guidance` for which representation to ask for, and see [tutorials/1_get_embeddings](../../../tutorials/1_get_embeddings) for worked commands. You will need to provide the following inputs:
* `--model_config`: the path to the model config file. Download the model config from [Hugging Face](https://huggingface.co/ada-f/ATOMICA).
* `--model_weights`: the path to the model weights file. Download the model weights from [Hugging Face](https://huggingface.co/ada-f/ATOMICA).
* `--data_path`: the path to the processed data file from above, in pickle, parquet or json format.
* `--representations`: which representations to extract, comma separated. One of `h_atom`, `h_block`, `h_graph`, `h_interface`, `z_atom`, `z_block`, `z_graph`, `z_interface`.
* `--pool`: required for `z_graph` and `z_interface`. `mean_std_global` when a head will be trained on the vectors, `mean_component_normalized` when they will be compared directly.
* `--segment`: required for `h_interface` and `z_interface`. 0 is the first interface and 1 the second.
* `--output_path`: the path to the output file, in parquet or pickle format. The output has one row per molecular complex, with the keys `id`, `block_id`, `atom_id` and one key per requested representation, named after that representation.