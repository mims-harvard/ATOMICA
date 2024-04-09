from openbabel import openbabel
from openbabel import pybel
import os
import torch
import numpy as np
from openeye import oechem
import shutil

from data.pdb_utils import VOCAB

def generated_to_xyz(data):
    num_atoms, atom_type, atom_coords = data
    if type(atom_coords) == torch.Tensor:
        atom_coords = atom_coords.clone().cpu().tolist()
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = atom_type[i]
        x, y, z = atom_coords[i]
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)
    return xyz

def generated_to_sdf(data):
    xyz = generated_to_xyz(data)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz)
    sdf = obConversion.WriteString(mol)
    return sdf, mol

def visualize_mol_sdf(node_type, coords, save_sdf_path=None):
    num_atoms = len(node_type)
    data = (num_atoms, node_type, coords)
    sdf_string, ob_mol = generated_to_sdf(data)
    if save_sdf_path is None:
        return ob_mol
    with open(save_sdf_path, 'w') as f:
        f.write(sdf_string)
    return ob_mol

def visualize_data_sdf(data, save_dir=None):
    ob_mols = {}

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    segment_start = 0
    block_segment_ids = np.array(data["segment_ids"])
    block_len = np.array(data["block_lengths"])
    for segment_id in sorted(set(data['segment_ids'])):
        segment_len = block_len[block_segment_ids == segment_id].sum()
        segment_end = segment_start + segment_len
        if VOCAB.idx_to_atom(data['A'][segment_start]) == VOCAB.atom_global:
            segment_start += 1
        atom_types = [VOCAB.idx_to_atom(x) for x in data['A'][segment_start:segment_end]]
        coords = data["X"][segment_start:segment_end]
        ob_mol = visualize_mol_sdf(atom_types, coords, f"{save_dir}/segment_{segment_id}.sdf" if save_dir is not None else None)
        segment_start = segment_end
        ob_mols[segment_id] = ob_mol
    return ob_mols


def addh_to_mol(input_sdf_path, output_sdf_path):
    # Using OpenEye to add hydrogens, OpenBabel has this functionality but it does not work for some reason
    
    # Create an input stream for reading the SDF file
    ifs = oechem.oemolistream()
    if not ifs.open(input_sdf_path):
        print(f"Could not open {input_sdf_path} for reading.")
        exit()

    # Create an output stream for writing the modified SDF file
    ofs = oechem.oemolostream()
    if not ofs.open(output_sdf_path):
        print(f"Could not open {output_sdf_path} for writing.")
        exit()

    # Create a molecule object to store each molecule
    mol = oechem.OEGraphMol()

    # Loop through the molecules in the SDF file
    while oechem.OEReadMolecule(ifs, mol):
        # Add hydrogens to the molecule
        oechem.OEAssignImplicitHydrogens(mol)
        oechem.OEAddExplicitHydrogens(mol)
        
        # Write the molecule with added hydrogens to the output file
        oechem.OEWriteMolecule(ofs, mol)

    # Close the input and output streams
    ifs.close()
    ofs.close()

    print("Hydrogens have been added with OpenEye.")


def build_pybel_mols(data, with_hydrogens=True):
    if os.path.exists("./tmp_sdf"):
        raise Exception("./tmp_sdf directory already exists")
    
    ob_mols = visualize_data_sdf(data, "./tmp_sdf")
    pybel_mols = {}
    if not with_hydrogens:
        for k, v in ob_mols.items():
            pybel_mols[k] = pybel.Molecule(v)
    else:
        for segment_id in ob_mols:
            input_sdf_path = f"./tmp_sdf/segment_{segment_id}.sdf"
            output_sdf_path = f"./tmp_sdf/segment_{segment_id}h.sdf"
            addh_to_mol(input_sdf_path, output_sdf_path)
            for molecule in pybel.readfile("sdf",output_sdf_path):
                molecule.OBMol.AddNewHydrogens(0, True, 7.4)
                pybel_mols[segment_id] = molecule
    shutil.rmtree("./tmp_sdf")
    return pybel_mols

