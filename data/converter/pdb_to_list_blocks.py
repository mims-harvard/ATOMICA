#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Optional

from Bio.PDB import PDBParser

from data.dataset import Block, Atom, VOCAB


def pdb_to_list_blocks(pdb: str, selected_chains: Optional[List[str]]=None, return_indexes=False, is_rna=False, is_dna=False) -> List[List[Block]]:
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
    '''

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('anonym', pdb)

    list_blocks, list_indexes, chain_ids = [], [], {}

    for chain in structure.get_chains():

        _id = chain.get_id()
        if (selected_chains is not None) and (_id not in selected_chains):
            continue

        residues, indexes, res_ids = [], [], {}

        for residue in chain:
            abrv = residue.get_resname()
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
            atoms = [ Atom(atom.get_id(), atom.get_coord(), atom.element) for atom in residue if atom.element != 'H' ]
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
        list_blocks = [list_blocks[chain_ids[chain_id]] for chain_id in selected_chains]
        list_indexes = [list_indexes[chain_ids[chain_id]] for chain_id in selected_chains]
    
    if return_indexes:
        return list_blocks, list_indexes
    return list_blocks



if __name__ == '__main__':
    import sys
    list_blocks = pdb_to_list_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(list_blocks)}')
    for i, chain in enumerate(list_blocks):
        print(f'chain {i} lengths: {len(chain)}')