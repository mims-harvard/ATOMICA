# Source https://github.com/THUNLP-MT/GET

import os
import sys
from copy import deepcopy
from typing import Tuple, List, Optional

from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from .mol_bpe import Tokenizer as PSTokenizer

from .mol_atom_match import struct_to_bonds
from .chem_utils import mol2smi, smi2mol, MAX_VALENCE
from .singleton import singleton


ID2BOND = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


@singleton
class TokenizerWrapper:

    def __init__(self, method=None):
        self.tokenizer = self.load(method)

    def load(self, method: Optional[str]):
        if method is None:
            return
        abs_base_path = os.path.dirname(os.path.abspath(__file__))
        if method == 'PS_300':
            self.tokenizer = PSTokenizer(os.path.join(abs_base_path, 'vocabs', 'ps_vocab_300.txt'))
        elif method == 'PS_500':
            self.tokenizer = PSTokenizer(os.path.join(abs_base_path, 'vocabs', 'ps_vocab_500.txt'))
        else:
            raise ValueError('Valid fragmentation method not found')
        
    def __call__(self, mol):
        return self.tokenizer(mol)
    
    def get_frag_smiles(self):
        frags = []
        for i in range(len(self.tokenizer)):
            smi = self.tokenizer.idx_to_subgraph(i)
            frags.append(smi)
        return frags
    
    def __len__(self):
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)


TOKENIZER = TokenizerWrapper()


def clean_bonds(atoms: List[str], bonds: List[Tuple[int, int, int]]):
    valence = { i: 0 for i in range(len(atoms)) }
    atom2bonds = { i: [] for i in range(len(atoms)) }
    for i, bond in enumerate(bonds):
        src, dst, _type = bond
        if _type == 4:
            val = 1.5 # aromatic, two aromatic bonds forms 3 valence
        else:
            val = _type
        valence[src] += val
        valence[dst] += val
        atom2bonds[src].append(i)
        atom2bonds[dst].append(i)
    
    bonds = deepcopy(bonds)
    
    # 1. clean wrong end-atom aromatic e.g. C(=O)O
    for i in valence:
        if int(valence[i]) != valence[i]: # end atom is marked aromatic (e.g. O outside the ring)
            if atoms[i] == 'C':
                continue
            if atoms[i] == 'N' and valence[i] == 4.5:
                for j in atom2bonds[i]:
                    bond = bonds[j]
                    bonds[j] = (bond[0], bond[1], 1)
                    src, dst = bond[0], bond[1]
                    if src != i:
                        src, dst = dst, src
                    valence[src] += 0.5
                    valence[dst] += 0.5
                continue

            bond, bond_idx = None, None
            for j in atom2bonds[i]:
                bond = bonds[j]
                if bond[-1] == 4:
                    bond_idx = j
                    break
            bonds[bond_idx] = (bond[0], bond[1], 2) # double bond
            src, dst = bond[0], bond[1]
            if src != i:
                src, dst = dst, src
            valence[src] += 0.5
            valence[dst] += 0.5

    for i in valence:
        if valence[i] > MAX_VALENCE[atoms[i]] and (atoms[i] == 'C' or atoms[i] == 'S'):
            # valence tautomerism
            while valence[i] > MAX_VALENCE[atoms[i]]:
                minused = False
                for j in atom2bonds[i]:
                    bond = bonds[j]
                    if bond[-1] > 1 and bond[-1] < 4: # double / triple bond
                        bonds[j] = (bond[0], bond[1], bond[2] - 1)
                        valence[bond[0]] -= 1
                        valence[bond[1]] -= 1
                        minused = True
                        break
                if not minused: # no bond can be reduced
                    break
    return bonds


def format_atom(atom):
    if len(atom) == 2:
        atom = atom[0].upper() + atom[1].lower()
    return atom


def tokenize_3d(
        atoms: List[str],
        coords: Optional[List[Tuple[float, float, float]]]=None,
        smiles: Optional[str]=None,
        bonds: Optional[List[Tuple[int, int, int]]]=None,
        fragmentation_method: Optional[str]=None
    ):
    
    tokenizer = TOKENIZER
    tokenizer.load(fragmentation_method)
    assert (coords is not None and smiles is not None) or (bonds is not None)
    atoms = [format_atom(atom) for atom in atoms]
    rw_mol = Chem.RWMol()
    for symbol in atoms:
        new_atom = Chem.Atom(symbol)
        rw_mol.AddAtom(new_atom)

    # print({ i: a for i, a in enumerate(atoms) })
    if bonds is None:
        # print(smiles)
        bonds = struct_to_bonds(atoms, coords, smiles)
    # print(bonds)

    bonds = clean_bonds(atoms, bonds)
    # print(bonds)
    for src, dst, _type in bonds:
        rw_mol.AddBond(src, dst, ID2BOND[_type])

    # add formal charge on N+
    rw_mol.UpdatePropertyCache(strict=False)
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() == 4 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(1)

    new_mol = rw_mol.GetMol()
    # print(mol2smi(new_mol))
    Chem.SanitizeMol(new_mol) # add aromatic bonds
    # print([new_mol.GetAtomWithIdx(i).GetSymbol() for i in range(new_mol.GetNumAtoms())])
    # print([bond.GetBondType() for bond in new_mol.GetBonds()])
    # print([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in new_mol.GetBonds()])
    # print([bond.GetEndAtomIdx() for bond in new_mol.GetBonds()])

    frag_mol = tokenizer(new_mol)
    frags, atom_idxs = [], []
    for i in frag_mol:
        node = frag_mol.get_node(i)
        frags.append(node.smiles)
        atom_idxs.append(list(node.atom_mapping.keys()))

    return frags, atom_idxs
