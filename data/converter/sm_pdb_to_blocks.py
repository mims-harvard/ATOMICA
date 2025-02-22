# Source https://github.com/THUNLP-MT/GET

from rdkit import Chem
import os
from data.dataset import Block, Atom, VOCAB
from .atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from rdkit.Chem.rdchem import GetPeriodicTable
_periodic_table = GetPeriodicTable()

def sm_pdb_to_blocks(ligand_path, fragment=None, mol_idx=0):
    if ligand_path.endswith(".pdb"):
        pdb_mol = Chem.MolFromPDBFile(ligand_path, removeHs=True, sanitize=False)
    else:
        pdb_mol = Chem.SDMolSupplier(ligand_path, removeHs=True, sanitize=False)[mol_idx]
    # ligand (each block is an atom)
    blocks2 = []
    conf = pdb_mol.GetConformer()
    ligand_coords = conf.GetPositions()
    for atom, atom_coords in zip(pdb_mol.GetAtoms(), ligand_coords):
        coords = atom_coords.tolist()
        atom_index = atom.GetAtomicNum()
        element = _periodic_table.GetElementSymbol(atom_index)
        atom_name = element
        # input(element)
        atom = Atom(
            atom_name=atom_name,
            # e.g. C1, C2, ..., these position code will be a unified encoding such as <sm> (small molecule) in our framework
            coordinate=coords,
            element=element,
            pos_code=VOCAB.atom_pos_sm
        )
        blocks2.append(Block(
            symbol=atom.element.lower(),
            units=[atom]
        ))
    
    if fragment is not None:
        # bonds
        bonds = []
        for bond in pdb_mol.GetBonds():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_name = str(bond.GetBondType())
            if bond_name == 'SINGLE':
                _type = 1
            elif bond_name == 'DOUBLE':
                _type = 2
            elif bond_name == 'TRIPLE':
                _type = 3
            elif bond_name == 'AROMATIC':
                _type = 4
            else:
                continue
            bonds.append((src, dst, _type))
        blocks2 = atom_blocks_to_frag_blocks(blocks2, bonds=bonds, fragmentation_method=fragment)
    return blocks2