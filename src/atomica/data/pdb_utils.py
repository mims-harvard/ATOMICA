# Source https://github.com/THUNLP-MT/GET

from copy import copy, deepcopy
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom

from .tokenizer.tokenize_3d import TOKENIZER


BACKBONE = ['N', 'CA', 'C', 'O']

SIDECHAIN = {
    'G': [],   # -H
    'A': ['CB'],  # -CH3
    'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
    'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
    'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
    'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
    'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
    'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
    'S': ['CB', 'OG'],  # -CH2-OH
    'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
    'C': ['CB', 'SG'],  # -CH2-SH
    'P': ['CB', 'CG', 'CD'],  # -C3H6
}

ATOMS = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]


def format_atom_element(atom: str):
    atom = atom.upper()
    if len(atom) == 2:
        atom = atom[0] + atom[1].lower()
    return atom


class Vocab:

    def __init__(self):
        self.PAD, self.MASK, self.UNK = '#', '*', '?'
        self.GLB = '&'  # global node
        specials = [# special added
                (self.PAD, 'PAD'), (self.MASK, 'MASK'), (self.UNK, 'UNK'), # pad / mask / unk
                (self.GLB, '<G>')  # global node
            ]
        # specials = []
        # (symbol, abbrv)
        aas = [  # amino acids
                ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
                ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
                ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
                ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
                ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO') # 20 aa
                # ('U', 'SEC') # 21 aa for eukaryote
            ]

        bases = [ # bases for RNA/DNA
                ('DA', 'DA'), ('DG', 'DG'), ('DC', 'DC'), ('DT', 'DT'), # DNA
                ('RA', 'RA'), ('RG', 'RG'), ('RC', 'RC'), ('RU', 'RU'), # RNA, RI = inosine
        ]

        sms = [(atom.lower(), atom) for atom in ATOMS]
        
        frags = [] # principal subgraphs
        if len(TOKENIZER):
            _tmp_map = { atom: True for atom in ATOMS }
            for i, smi in enumerate(TOKENIZER.get_frag_smiles()):
                if smi in _tmp_map: # single atom
                    continue
                frags.append((str(i), smi))
        
        self.aas, self.bases, self.sms, self.frags = aas, bases, sms, frags

        self.atom_pad, self.atom_mask, self.atom_global = 'p', 'm', 'g'
        self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_global = 'p', 'm', 'g'
        self.atom_pos_sm = 'sm'  # small molecule


        _all = specials + aas + bases + sms + frags
        self.symbol2idx, self.abrv2idx = {}, {}
        self.idx2symbol, self.idx2abrv = [], []
        for i, (symbol, abrv) in enumerate(_all):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
            self.idx2symbol.append(symbol)
            self.idx2abrv.append(abrv)
        self.special_mask = [1 for _ in specials] + [0 for _ in aas + bases + sms + frags]
        assert len(self.symbol2idx) == len(self.idx2symbol)
        assert len(self.abrv2idx) == len(self.idx2abrv)
        assert len(self.idx2symbol) == len(self.idx2abrv)

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, self.atom_global] + ATOMS
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_global, ''] + \
                            ['A', 'B', 'G', 'D', 'E', 'Z', 'H', 'XT', 'P'] + \
                            [self.atom_pos_sm] + ["'"]  # SM is for atoms in small molecule, 'P' for O1P, O2P, O3P, "'" for bases
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    
    def load_tokenizer(self, method: Optional[str]):
        if method is None:
            return
        TOKENIZER.load(method)
        self.__init__()

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx_to_symbol(idx)

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2abrv[idx]

    def abrv_to_idx(self, abrv):
        return self.abrv2idx.get(abrv, self.abrv2idx['UNK'])

    def symbol_to_idx(self, symbol):
        return self.symbol2idx.get(symbol, self.abrv2idx['UNK'])
    
    def idx_to_symbol(self, idx):
        return self.idx2symbol[idx]

    def idx_to_abrv(self, idx):
        return self.idx2abrv[idx]

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_global_idx(self):
        return self.atom2idx[self.atom_global]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def get_atom_pos_global_idx(self):
        return self.atom_pos2idx[self.atom_pos_global]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx.get(atom, self.atom2idx[self.atom_mask])

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx.get(atom_pos, self.atom_pos2idx[self.atom_pos_mask])

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = Vocab()


def format_aa_abrv(abrv):  # special cases
    if abrv == 'MSE':
        return 'MET' # substitue MSE with MET
    return abrv


class Atom:
    def __init__(self, atom_name: str, coordinate: List, element: str, pos_code: str=None):
        self.name = atom_name
        self.coordinate = coordinate
        self.element = format_atom_element(element)
        if pos_code is None:
            pos_code = atom_name.lstrip(element)
            pos_code = ''.join((c for c in pos_code if not c.isdigit()))
            self.pos_code = pos_code
        else:
            self.pos_code = pos_code

    def get_element(self):
        return self.element
    
    def get_coord(self):
        return copy(self.coordinate)
    
    def get_pos_code(self):
        return self.pos_code
    
    def __str__(self) -> str:
        return self.name


class Residue:
    def __init__(self, symbol: str, atom_map: Dict, _id: Tuple, sidechain: List=None):
        self.symbol = symbol
        self.atom_map = atom_map
        self.sidechain = sidechain
        if self.sidechain is None:
            if symbol in SIDECHAIN:
                self.sidechain = SIDECHAIN[symbol]
            else:
                self.sidechain = [atom for atom in atom_map if atom not in BACKBONE]
        self.id = _id  # (residue_number, insert_code), ' ' for null insert_code

    def get_symbol(self):
        return self.symbol

    def get_coord(self, atom_name):
        return copy(self.atom_map[atom_name].coordinate)

    def get_coord_map(self) -> Dict[str, List]:
        return { atom_name: copy(self.atom_map[atom_name].coordinate) for atom_name in self.atom_map}

    def get_backbone_coord_map(self) -> Dict[str, List]:
        return { atom_name: copy(self.atom_map[atom_name].coordinate) for atom_name in self.atom_map if atom_name in BACKBONE}

    def get_sidechain_coord_map(self) -> Dict[str, List]:
        coord = {}
        for atom in self.sidechain:
            if atom in self.atom_map:
                coord[atom] = copy(self.atom_map[atom].coordinate)
        return coord

    def get_atom_names(self):
        return list(self.atom_map.keys())
    
    def get_atom(self, atom_name):
        return deepcopy(self.atom_map[atom_name])

    def get_id(self):
        return self.id

    def has_atom(self, atom_name):
        return atom_name in self.atom_map

    def set_symbol(self, symbol):
        assert VOCAB.symbol_to_abrv(symbol) is not None, f'{symbol} is not an amino acid'
        self.symbol = symbol

    def set_atom_map(self, atom_map):
        self.atom_map = deepcopy(atom_map)

    def dist_to(self, residue):  # measured by nearest atoms
        xa = np.array(list(self.get_coord_map().values()))
        xb = np.array(list(residue.get_coord_map().values()))
        if len(xa) == 0 or len(xb) == 0:
            return math.nan
        dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        return np.min(dist)

    def to_bio(self):
        _id = (' ', self.id[0], self.id[1])
        abrv = self.real_abrv if hasattr(self, 'real_abrv') else VOCAB.symbol_to_abrv(self.symbol)
        residue = BResidue(_id, abrv, '    ')
        atom_map = self.atom_map
        for i, atom_name in enumerate(atom_map):
            atom = atom_map[atom_name]
            fullname = ' ' + atom_name
            while len(fullname) < 4:
                fullname += ' '
            bio_atom = BAtom(
                name=atom_name,
                coord=np.array(atom.coordinate, dtype=np.float32),
                bfactor=0,
                occupancy=1.0,
                altloc=' ',
                fullname=fullname,
                serial_number=i,
                element=atom.element
            )
            residue.add(bio_atom)
        return residue

    def __iter__(self):
        return iter([(atom_name, self.atom_map[atom_name]) for atom_name in self.atom_map])
    
    def __len__(self):
        return len(self.atom_map)


class Peptide:
    def __init__(self, _id, residues: List[Residue]):
        self.residues = residues
        self.seq = ''
        self.id = _id
        for residue in residues:
            self.seq += residue.get_symbol()

    def set_id(self, _id):
        self.id = _id

    def get_id(self):
        return self.id

    def get_seq(self):
        return self.seq

    def get_span(self, i, j):  # [i, j)
        i, j = max(i, 0), min(j, len(self.seq))
        if j <= i:
            return None
        else:
            residues = deepcopy(self.residues[i:j])
            return Peptide(self.id, residues)

    def get_residue(self, i):
        return deepcopy(self.residues[i])
    
    def get_ca_pos(self, i):
        return copy(self.residues[i].get_coord('CA'))

    def get_cb_pos(self, i):
        return copy(self.residues[i].get_coord('CB'))

    def set_residue_symbol(self, i, symbol):
        self.residues[i].set_symbol(symbol)
        self.seq = self.seq[:i] + symbol + self.seq[i+1:]

    def set_residue(self, i, symbol, coord):
        self.set_residue_symbol(i, symbol)
        self.set_residue_coord(i, coord)

    def to_bio(self):
        chain = BChain(id=self.id)
        for residue in self.residues:
            chain.add(residue.to_bio())
        return chain

    def __iter__(self):
        return iter(self.residues)

    def __len__(self):
        return len(self.residues)

    def __str__(self):
        return self.seq


class Protein:
    def __init__(self, pdb_id, peptides):
        self.pdb_id = pdb_id
        self.peptides = peptides

    @classmethod
    def from_pdb(cls, pdb_path, include_all=False):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper().strip()
        if pdb_id == '':
            # deduce from file name
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

        peptides = {}
        for chain in structure.get_chains():

            _id = chain.get_id()
            residues, unk_cnt, res_ids = [], 0, {}
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                res_id = f'{res_number}-{insert_code}'
                if hetero_flag == 'W':
                    continue   # residue from glucose (WAT) or water (HOH)
                if hetero_flag.strip() != '' and res_id in res_ids and not include_all:
                    continue  # the solution (e.g. H_EDO (EDO))
                if abrv == 'MSE':
                    abrv = 'MET'  # MET is usually transformed to MSE for structural analysis
                if abrv in ['A', 'G', 'C', 'U']:
                    abrv = 'R' + abrv # RNA
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol == VOCAB.UNK:
                    # print(abrv)
                    sidechain = [atom.get_id() for atom in residue if atom.element != 'H']
                    sidechain = [atom for atom in sidechain if atom not in BACKBONE]
                    unk_cnt += 1
                else:
                    sidechain = None # automatically decided
                    
                # filter Hs because not all data include them
                atoms = { atom.get_id(): Atom(atom.get_id(), atom.get_coord(), atom.element) for atom in residue if atom.element != 'H' }
                sidechain = [atom.get_id() for atom in residue if atom.element != 'H']
                sidechain = [atom for atom in sidechain if atom not in BACKBONE]
                residues.append(Residue(
                    symbol, atoms, (res_number, insert_code), sidechain
                ))
                res_ids[res_id] = True
                if symbol == VOCAB.UNK:
                    residues[-1].real_abrv = abrv
            
            # the last few residues might be solvents
            end = len(residues) - 1
            while end >= 0 and not include_all:
                if residues[end].symbol == VOCAB.UNK:
                    end -= 1
                else:
                    break
            residues = residues[:end + 1]
            if len(residues) == 0:  # not a peptide
                continue

            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)

    def get_id(self):
        return self.pdb_id

    def num_chains(self):
        return len(self.peptides)

    def get_chain(self, name):
        if name in self.peptides:
            return deepcopy(self.peptides[name])
        else:
            return None

    def get_chain_names(self):
        return list(self.peptides.keys())

    def to_bio(self):
        structure = BStructure(id=self.pdb_id)
        model = BModel(id=0)
        for name in self.peptides:
            model.add(self.peptides[name].to_bio())
        structure.add(model)
        return structure

    def to_pdb(self, path, atoms=None):
        if atoms is None:
            bio_structure = self.to_bio()
        else:
            prot = deepcopy(self)
            for _, chain in prot:
                for residue in chain:
                    atom_map = {}
                    res_atoms = residue.get_atom_names()
                    for atom in atoms:
                        if atom in res_atoms:
                            atom_map[atom] = residue.get_atom(atom)
                    residue.set_atom_map(atom_map)
            bio_structure = prot.to_bio()
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(path)

    def __iter__(self):
        return iter([(c, self.peptides[c]) for c in self.peptides])

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError('Cannot compare other type to Protein')
        for key in self.peptides:
            if key in other.peptides and self.peptides[key].seq == other.peptides[key].seq:
                continue
            else:
                return False
        return True

    def __str__(self):
        res = self.pdb_id + '\n'
        for seq_name in self.peptides:
            res += f'\t{seq_name}: {self.peptides[seq_name]}\n'
        return res


class Complex(Protein):
    def __init__(self, pdb_id, peptides, receptor_chains=None, ligand_chains=None):
        super().__init__(pdb_id, peptides)
        assert not (receptor_chains is None and ligand_chains is None), 'At least one of receptor_chains or ligand_chains should be provided'
        if type(receptor_chains) == str:
            receptor_chains = list(receptor_chains)
        if type(ligand_chains) == str:
            ligand_chains = list(ligand_chains)
        if receptor_chains is None:
            self.ligand_chains = ligand_chains
            self.receptor_chains = [chain for chain in self.peptides if chain not in self.ligand_chains]
        elif ligand_chains is None:
            self.receptor_chains = receptor_chains
            self.ligand_chains = [chain for chain in self.peptides if chain not in self.receptor_chains]
        else:
            self.receptor_chains = receptor_chains
            self.ligand_chains = ligand_chains

    @classmethod
    def from_pdb(cls, pdb_path, receptor_chains=None, ligand_chains=None, include_all=False):
        prot = Protein.from_pdb(pdb_path, include_all)
        return cls(prot.pdb_id, prot.peptides, receptor_chains, ligand_chains)

    def get_interacting_residues(self, dist_th=10):
        '''
        calculate interacting residues based on minimum distance between heavy atoms < 10A (default)
        '''
        rec_residues, lig_residues = [], []
        for res_list, chains in zip([rec_residues, lig_residues], [self.receptor_chains, self.ligand_chains]):
            for chain in chains:
                for residue in self.peptides[chain]:
                    res_list.append((chain, residue))
        # calculate distance
        dist = dist_matrix_from_residues(
            [tup[1] for tup in rec_residues],
            [tup[1] for tup in lig_residues]
        )  # [Nrec, Nlig]
        is_interacting = dist < dist_th
        rec_index = np.nonzero(is_interacting.sum(axis=1) > 0)[0]
        lig_index = np.nonzero(is_interacting.sum(axis=0) > 0)[0]

        rec_inter = [rec_residues[i] for i in rec_index]
        lig_inter = [lig_residues[i] for i in lig_index]
        return rec_inter, lig_inter

    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        ligand_info = f'Ligand Chain: {[(chain_name, len(self.get_chain(chain_name))) for chain_name in self.ligand_chains]}'
        receptor_info = f'Receptor Chains: {[(chain_name, len(self.get_chain(chain_name))) for chain_name in self.receptor_chains]}'
        epitope_info = f'Epitope: \n'
        # residue_map = {}
        # for _, chain_name, i in self.get_epitope():
        #     if chain_name not in residue_map:
        #         residue_map[chain_name] = []
        #     residue_map[chain_name].append(i)
        # for chain_name in residue_map:
        #     epitope_info += f'\t{chain_name}: {sorted(residue_map[chain_name])}\n'

        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + ligand_info + '\n' + receptor_info + '\n' + epitope_info + sep


def atom_features_from_residues(residue_list):
    atoms_list, atom_positions_list = [], []
    for residue in residue_list:
        atoms, atom_positions = [], []
        for atom in BACKBONE:
            atoms.append(VOCAB.atom_to_idx(atom[0]))
            atom_positions.append(VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb))
        for atom in residue.sidechain:
            # print(atom)
            atoms.append(VOCAB.atom_to_idx(atom[0]))
            if len(atom) == 1:
                atom_positions.append(VOCAB.atom_pos_to_idx(VOCAB.atom_pos_mask))
            else:
                atom_positions.append(VOCAB.atom_pos_to_idx(atom[1]))
        num_pad = VOCAB.MAX_ATOM_NUMBER - len(atoms)
        atoms.extend([VOCAB.get_atom_pad_idx() for _ in range(num_pad)])
        atom_positions.extend([VOCAB.get_atom_pos_pad_idx() for _ in range(num_pad)])
        atoms_list.append(atoms)
        atom_positions_list.append(atom_positions)
    return atoms_list, atom_positions_list


def coords_from_residues(residue_list):
    coords, masks = [], []
    max_len = 0
    for residue in residue_list:
        x, mask = [], []
        coordinates = residue.get_coord_map()
        for atom in BACKBONE + residue.sidechain:
            if atom in coordinates:
                x.append(coordinates[atom])
                mask.append(1)
            else:
                x.append([0, 0, 0])
                mask.append(0)
        
        max_len = max(max_len, len(x))

        coords.append(x)
        masks.append(mask)
    
    for i in range(len(coords)):
        num_pad =  max_len - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == MAX_ATOM_NUM, in mask 0 for padding


def dist_matrix_from_coords(coords1, masks1, coords2, masks2):
    dist = np.linalg.norm(coords1[:, None] - coords2[None, :], axis=-1)  # [N1, N2, M]
    dist = dist + np.logical_not(masks1[:, None] * masks2[None, :]) * 1e6  # [N1, N2, M]
    dist = np.min(dist, axis=-1)  # [N1, N2]
    return dist


def dist_matrix_from_residues(residue_list1, residue_list2):
    coords, mask = coords_from_residues(residue_list1 + residue_list2)
    midpoint = len(residue_list1)
    coords1, masks1 = coords[:midpoint], mask[:midpoint]
    coords2, masks2 = coords[midpoint:], mask[midpoint:]
    return dist_matrix_from_coords(coords1, masks1, coords2, masks2)