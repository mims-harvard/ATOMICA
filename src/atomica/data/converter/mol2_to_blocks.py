# Source https://github.com/THUNLP-MT/GET

import re
from typing import List, Optional

from data.dataset import Block, Atom, VOCAB

from .atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks


def mol2_to_blocks(mol2_file: str, using_hydrogen: bool = False, molecule_type: Optional[str] = None, fragment: str=None) -> List[Block]:
    '''
        Convert an Mol2 file to a list of lists of blocks for each molecule / residue.
        
        Parameters:
            mol2_file: Path to the Mol2 file
            using_hydrogen: Whether to preserve hydrogen atoms, default false
            molecule_type: "protein" or "small" (small molecule). If not specified, deduce from the mol2 file

        Returns:
            A list of blocks reprensenting a small molecule / protein, etc.
    '''
    # Read Mol2 file
    with open(mol2_file, 'r') as fin:
        lines = fin.readlines()
    
    # extract molecule information and atom information
    molecule_infos, atom_infos = [], []
    infos, bond_start = None, -1
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            continue

        if line == '@<TRIPOS>MOLECULE':
            infos = molecule_infos
        elif line == '@<TRIPOS>ATOM':
            infos = atom_infos
        elif line == '@<TRIPOS>BOND':
            bond_start = i
            break
        elif line.startswith('@<TRIPOS>'):  # other sections
            infos = None
        elif infos is None:
            continue # still in file head
        else:
            infos.append(line)

    # protein or small molecule
    if molecule_type is None:
        if 'PROTEIN' in molecule_infos or 'BIOPOLYMER' in molecule_infos:
            molecule_type = 'protein'
        elif 'SMALL' in molecule_infos:
            molecule_type = 'small'
        else:
            raise ValueError('Molecule type not specified in the head')

    def line_to_atom(line):
        _, name, x, y, z, element, res_id, res_name, _ = re.split(r'\s+', line)[:9]
        element = element.split('.')[0]
        atom = Atom(name, [float(x), float(y), float(z)], element)
        return atom, res_id, res_name

    # to list of blocks
    blocks = []

    if molecule_type == 'small':
        remap = {}
        for i, line in enumerate(atom_infos):
            atom, _, _ = line_to_atom(line)
            if not using_hydrogen and atom.element == 'H':
                continue
            blocks.append(Block(atom.element.lower(), [atom]))
            remap[i + 1] = len(remap) # atom indexes in the records start from 1
        if fragment is not None:
            bonds = []
            for line in lines[bond_start + 1:]:
                if line.startswith('@'):
                    break
                _, src, dst, _type = re.split(r'\s+', line.strip())
                if _type.isdigit():
                    _type = int(_type)
                elif _type == 'ar': # aromatic
                    _type = 4
                elif _type == 'am': # amide
                    _type = 1
                elif _type in ['du', 'un', 'nc']:
                    continue
                else:
                    raise ValueError(f'bond type {_type} not recognized!')
                src, dst = int(src), int(dst)
                if src not in remap or dst not in remap:
                    continue
                bonds.append((remap[src], remap[dst], _type))
            blocks = atom_blocks_to_frag_blocks(blocks, bonds=bonds, fragmentation_method=fragment)
    elif molecule_type == 'protein':
        residues = {}
        for line in atom_infos:
            atom, res_id, res_name = line_to_atom(line)
            if not using_hydrogen and atom.element == 'H':
                continue
            if res_id not in residues:
                residues[res_id] = {'resname': res_name, 'atoms': []}
            residues[res_id]['atoms'].append(atom)
        for res_id in residues:
            residue = residues[res_id]
            resname = ''
            for char in residue['resname']:
                if char.isdigit():
                    continue
                resname += resname
            blocks.append(Block(
                VOCAB.abrv_to_symbol(resname.upper()),
                residue['atoms']
            ))

    else:
        raise NotImplementedError(f'Molecule type {molecule_type} not implemented')
        

    # Return the final list of lists of blocks
    return blocks

if __name__ == '__main__':
    import sys
    list_blocks = mol2_to_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of blocks: {len(list_blocks)}')
