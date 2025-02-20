# Source https://github.com/THUNLP-MT/GET

from typing import List, Optional, Dict, Tuple
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
import biotite.structure as bs
from biotite.structure import AtomArray, get_residue_starts
from biotite.structure.io.pdb import PDBFile

import sys
import os
PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.dataset import Block, Atom, VOCAB


def pdb_to_list_blocks(pdb: str, selected_chains: Optional[List[str]]=None, 
                       return_indexes: bool =False, is_rna: bool=False, is_dna: bool=False, 
                       use_model:int =None) -> Tuple[List[List[Block]], Optional[Dict[str, int]]]:
    '''
        Convert pdb file to a list of lists of blocks using Biopython.
        Each chain will be a list of blocks.
        
        Parameters:
            pdb: Path to the pdb file
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']

        Returns:
            A list of lists of blocks. Each chain in the pdb file will be parsed into
            one list of blocks.
            example:
                [
                    [residueA1, residueA2, ...],  # chain A
                    [residueB1, residueB2, ...]   # chain B
                ],
                where each residue is instantiated by Block data class.
            
            If return_indexes, also returns a list of residue indexes for each chain. 
            Each residue is indexed with the format "<chain_id>_<residue_number>".
    '''
    if pdb.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb)
    elif pdb.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb)
    else:
        raise ValueError(f"Unsupported PDB file type, {pdb}")

    list_blocks, list_indexes, chain_ids = [], [], {}
    
    if use_model is not None:
        structure = structure[use_model]

    for chain in structure.get_chains():

        _id = chain.get_id()
        if (selected_chains is not None) and (_id not in selected_chains):
            continue

        residues, indexes, res_ids = [], [], {}

        for residue in chain:
            abrv = residue.get_resname().strip()
            hetero_flag, res_number, insert_code = residue.get_id()
            res_id = f'{res_number}-{insert_code}'
            if hetero_flag == 'W':
                continue   # residue from glucose (WAT) or water (HOH)
            if hetero_flag.strip() != '' and res_id in res_ids:
                continue  # the solution (e.g. H_EDO (EDO))
            if abrv == 'MSE':
                abrv = 'MET'  # MET is usually transformed to MSE for structural analysis
            
            # some pdbs use single letter code for DNA and RNA
            if is_dna and abrv in {'A', 'T', 'G', 'C'} and not abrv.startswith("D"):
                abrv = "D" + abrv
            if is_rna and abrv in {'A', 'U', 'G', 'C'} and not abrv.startswith("R"):
                abrv = "R" + abrv
            symbol = VOCAB.abrv_to_symbol(abrv)
                
            # filter Hs because not all data include them
            atoms = [ Atom(atom.get_id(), atom.get_coord().tolist(), atom.element) for atom in residue if atom.element != 'H' ]
            residues.append(Block(symbol, atoms))
            res_ids[res_id] = True
            indexes.append(f"{_id}_{res_number}")
        
        # the last few residues might be non-relevant molecules in the solvent if their types are unk
        end = len(residues) - 1
        while end >= 0:
            if residues[end].symbol == VOCAB.UNK:
                end -= 1
            else:
                break
        residues = residues[:end + 1]
        indexes = indexes[:end + 1]
        if len(residues) == 0:  # not a chain
            continue

        chain_ids[_id] = len(list_blocks)
        list_blocks.append(residues)
        list_indexes.append(indexes)

    # reorder
    if selected_chains is not None:
        for chain_id in selected_chains:
            if chain_id not in chain_ids:
                raise ValueError(f"Chain {chain_id} not found in the PDB file {pdb}")
        list_blocks = [list_blocks[chain_ids[chain_id]] for chain_id in selected_chains]
        list_indexes = [list_indexes[chain_ids[chain_id]] for chain_id in selected_chains]
    
    if return_indexes:
        return list_blocks, list_indexes
    return list_blocks


def atoms_array_to_blocks(atoms_array: AtomArray) -> List[Block]:
    residue_starts = get_residue_starts(atoms_array)
    residue_starts = np.concatenate([residue_starts, [len(atoms_array)]])
    next_start = residue_starts[0]
    start_idx = 0
    curr_res_name = None
    atoms = []
    residues = []
    for atom_index, atom in enumerate(atoms_array):
        if atom_index == next_start:
            if len(atoms) > 0:
                symbol = VOCAB.abrv_to_symbol(curr_res_name)
                residues.append(Block(symbol, atoms))
            atoms = []
            start_idx += 1
            next_start = residue_starts[start_idx]
        curr_res_name = atom.res_name
        if atom.element == 'H':
            continue
        atoms.append(Atom(atom.atom_name, atom.coord.tolist(), atom.element))
    if len(atoms) > 0:
        symbol = VOCAB.abrv_to_symbol(curr_res_name)
        residues.append(Block(symbol, atoms))
    return residues


def get_residues(atoms_array: AtomArray) -> Tuple[np.ndarray, List[Tuple[str, int, str, str]]]:
    # residues: (chain_id, res_id, res_name, ins_code)
    residue_starts = get_residue_starts(atoms_array)
    residues = []
    for res_idx in residue_starts:
        residues.append((atoms_array.chain_id[res_idx], atoms_array.res_id[res_idx], atoms_array.res_name[res_idx], atoms_array.ins_code[res_idx]))
    return residue_starts, residues


def pdb_to_list_blocks_and_atom_array(pdb: str, selected_chains: Optional[List[str]]=None, 
                       is_rna: bool=False, is_dna: bool=False, 
                       use_model:int =None) -> Tuple[List[List[Block]], AtomArray, List[List[Tuple[str, int, str, str]]]]:
    try:
        pdb_file = PDBFile.read(pdb)
        atom_array = pdb_file.get_structure()[use_model if use_model is not None else 0]
    except Exception as e:
        print(f"Error reading pdb file {pdb}: {e}")
        return [], None, []
    if is_rna or is_dna:
        atom_array = atom_array[bs.filter_nucleotides(atom_array)]
        if is_rna:
            atom_array.res_name = np.array([f"R{res_name}" if res_name in {'A', 'U', 'G', 'C'} else res_name for res_name in atom_array.res_name])
        elif is_dna:
            atom_array.res_name = np.array([f"D{res_name}" if res_name in {'A', 'T', 'G', 'C'} else res_name for res_name in atom_array.res_name])
    else:
        atom_array = atom_array[bs.filter_amino_acids(atom_array)]
    if selected_chains is not None:
        atom_array = atom_array[np.isin(atom_array.chain_id, selected_chains)]
    
    list_blocks = []
    list_residues = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        _, residues = get_residues(chain_atom_array)
        blocks = atoms_array_to_blocks(chain_atom_array)
        list_blocks.append(blocks)
        list_residues.append(residues)
    return list_blocks, atom_array, list_residues


if __name__ == '__main__':
    import sys
    list_blocks, atom_array, list_residues = pdb_to_list_blocks_and_atom_array(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(list_blocks)}')
    for i, chain in enumerate(list_blocks):
        print(f'chain {i} lengths: {len(chain)}')
    
    print(f'number of residues: {len(atom_array)}')
    print(f'list of residues: {list_residues}')