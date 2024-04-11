from openbabel import pybel
import openbabel
from collections import namedtuple
from enum import Enum
from typing import Tuple
import numpy as np
import itertools

from plip.basic import config
from plip.basic.supplemental import normalize_vector, vector, ring_is_planar
from plip.basic.supplemental import centroid, whichrestype
from plip.structure.detection import halogen, pication, hydrophobic_interactions, pistacking, hbonds, metal_complexation, filter_contacts
from plip.basic.supplemental import vector, euclidean3d
from plip.basic.supplemental import whichresnumber, whichrestype, whichchain
from plip.structure.preparation import PLInteraction
from data.pdb_utils import VOCAB


class SegmentType(Enum):
    LIGAND = 0
    BINDING_SITE = 1

class Mol:
    def __init__(self, mapper, segment_id):
        self.Mapper = mapper
        self.segment_id = segment_id
        self.original_mol = mapper.pybel_mols[segment_id]

        # self.rings = self.find_rings(self.original_mol, self.original_mol.atoms)
        # self.hydroph_atoms = self.hydrophobic_atoms(self.original_mol.atoms)
        # self.charged = None
        # self.hbond_don_atom_pairs = self.find_hbd(self.original_mol.atoms, self.hydroph_atoms)
        # self.hbond_acc_atoms = self.find_hba(self.original_mol.atoms)

    def hydrophobic_atoms(self, all_atoms):
        """Select all carbon atoms which have only carbons and/or hydrogens as direct neighbors."""
        atom_set = []
        data = namedtuple('hydrophobic', 'atom orig_atom orig_idx')
        atm = [a for a in all_atoms if a.atomicnum == 6 and set([natom.GetAtomicNum() for natom
                                                                 in pybel.ob.OBAtomAtomIter(a.OBAtom)]).issubset(
            {1, 6})]
        for atom in atm:
            orig_idx = self.Mapper.mapid(self.segment_id, atom.idx)
            orig_atom = self.Mapper.id_to_atom(orig_idx)
            atom_set.append(data(atom=atom, orig_atom=orig_atom, orig_idx=orig_idx))
        return atom_set

    def find_hba(self, all_atoms):
        """Find all possible hydrogen bond acceptors"""
        data = namedtuple('hbondacceptor', 'a a_orig_atom a_orig_idx type')
        a_set = []
        for atom in filter(lambda at: at.OBAtom.IsHbondAcceptor(), all_atoms):
            if atom.atomicnum not in [9, 17, 35, 53]:  # Exclude halogen atoms
                a_orig_idx = self.Mapper.mapid(self.segment_id, atom.idx)
                a_orig_atom = self.Mapper.id_to_atom(a_orig_idx)
                a_set.append(data(a=atom, a_orig_atom=a_orig_atom, a_orig_idx=a_orig_idx, type='regular'))
        a_set = sorted(a_set, key=lambda x: x.a_orig_idx)
        return a_set

    def find_hbd(self, all_atoms, hydroph_atoms):
        """Find all possible strong and weak hydrogen bonds donors (all hydrophobic C-H pairings)"""
        donor_pairs = []
        data = namedtuple('hbonddonor', 'd d_orig_atom d_orig_idx h type')
        for donor in [a for a in all_atoms if a.OBAtom.IsHbondDonor()]:
            in_ring = False
            if not in_ring:
                for adj_atom in [a for a in pybel.ob.OBAtomAtomIter(donor.OBAtom) if a.IsHbondDonorH()]:
                    d_orig_idx = self.Mapper.mapid(self.segment_id, donor.idx)
                    d_orig_atom = self.Mapper.id_to_atom(d_orig_idx)
                    donor_pairs.append(data(d=donor, d_orig_atom=d_orig_atom, d_orig_idx=d_orig_idx,
                                            h=pybel.Atom(adj_atom), type='regular'))
        for carbon in hydroph_atoms:
            for adj_atom in [a for a in pybel.ob.OBAtomAtomIter(carbon.atom.OBAtom) if a.GetAtomicNum() == 1]:
                d_orig_idx = self.Mapper.mapid(self.segment_id, carbon.atom.idx)
                d_orig_atom = self.Mapper.id_to_atom(d_orig_idx)
                donor_pairs.append(data(d=carbon, d_orig_atom=d_orig_atom,
                                        d_orig_idx=d_orig_idx, h=pybel.Atom(adj_atom), type='weak'))
        donor_pairs = sorted(donor_pairs, key=lambda x: (x.d_orig_idx, x.h.idx))
        return donor_pairs

    def find_rings(self, mol, all_atoms):
        """Find rings and return only aromatic.
        Rings have to be sufficiently planar OR be detected by OpenBabel as aromatic."""
        data = namedtuple('aromatic_ring', 'atoms orig_atoms atoms_orig_idx normal obj center type')
        rings = []
        aromatic_amino = ['TYR', 'TRP', 'HIS', 'PHE']
        ring_candidates = mol.OBMol.GetSSSR()
        # logger.debug(f'number of aromatic ring candidates: {len(ring_candidates)}')
        # Check here first for ligand rings not being detected as aromatic by Babel and check for planarity
        for ring in ring_candidates:
            r_atoms = [a for a in all_atoms if ring.IsMember(a.OBAtom)]
            r_atoms = sorted(r_atoms, key=lambda x: x.idx)
            if 4 < len(r_atoms) <= 6:
                res = list(set([whichrestype(a) for a in r_atoms]))
                # re-sort ring atoms for only ligands, because HETATM numbering is not canonical in OpenBabel
                if res[0] == 'UNL':
                    ligand_orig_idx = [self.Mapper.mapid(self.segment_id, a.idx) for a in r_atoms]
                    sort_order = np.argsort(np.array(ligand_orig_idx))
                    r_atoms = [r_atoms[i] for i in sort_order]
                if ring.IsAromatic() or res[0] in aromatic_amino or ring_is_planar(ring, r_atoms):
                    # Causes segfault with OpenBabel 2.3.2, so deactivated
                    # typ = ring.GetType() if not ring.GetType() == '' else 'unknown'
                    # Alternative typing
                    ring_type = '%s-membered' % len(r_atoms)
                    ring_atms = [r_atoms[a].coords for a in [0, 2, 4]]  # Probe atoms for normals, assuming planarity
                    ringv1 = vector(ring_atms[0], ring_atms[1])
                    ringv2 = vector(ring_atms[2], ring_atms[0])
                    atoms_orig_idx = []
                    atoms_orig_idx = [self.Mapper.mapid(self.segment_id, r_atom.idx) for r_atom in r_atoms]
                    orig_atoms = [self.Mapper.id_to_atom(idx) for idx in atoms_orig_idx]
                    rings.append(data(atoms=r_atoms,
                                      orig_atoms=orig_atoms,
                                      atoms_orig_idx=atoms_orig_idx,
                                      normal=normalize_vector(np.cross(ringv1, ringv2)),
                                      obj=ring,
                                      center=centroid([ra.coords for ra in r_atoms]),
                                      type=ring_type))
        return rings

    def get_hydrophobic_atoms(self):
        return self.hydroph_atoms

    def get_hba(self):
        return self.hbond_acc_atoms

    def get_hbd(self):
        return [don_pair for don_pair in self.hbond_don_atom_pairs if don_pair.type == 'regular']

    def get_weak_hbd(self):
        return [don_pair for don_pair in self.hbond_don_atom_pairs if don_pair.type == 'weak']

    def get_pos_charged(self):
        return [charge for charge in self.charged if charge.type == 'positive']

    def get_neg_charged(self):
        return [charge for charge in self.charged if charge.type == 'negative']


class Ligand(Mol):
    def __init__(self, mapper, segment_id):
        Mol.__init__(self, mapper, segment_id)
        self.all_atoms = self.original_mol.atoms
        self.rings = self.find_rings(self.original_mol, self.all_atoms)
        self.hydroph_atoms = self.hydrophobic_atoms(self.all_atoms)
        self.hbond_acc_atoms = self.find_hba(self.all_atoms)
        self.num_rings = len(self.rings)

        self.hbond_don_atom_pairs = self.find_hbd(self.all_atoms, self.hydroph_atoms)
        self.charged = self.find_charged(self.all_atoms)
        self.halogenbond_don = self.find_hal_don(self.all_atoms)
        self.halogenbond_acc = self.find_hal_acc(self.all_atoms)
        self.metals = []
        data = namedtuple('metal', 'm orig_m m_orig_idx')
        for a in [a for a in self.all_atoms if a.type.upper() in config.METAL_IONS]:
            m_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
            orig_m = self.Mapper.id_to_atom(m_orig_idx)
            self.metals.append(data(m=a, m_orig_idx=m_orig_idx, orig_m=orig_m))
        self.metal_binding = self.find_metal_binding(self.all_atoms)
        self.num_hba, self.num_hbd = len(self.hbond_acc_atoms), len(self.hbond_don_atom_pairs)
        self.num_hal = len(self.halogenbond_don)

    def get_canonical_num(self, atomnum):
        """Converts internal atom ID into canonical atom ID. Agrees with Canonical SMILES in XML."""
        return self.atomorder[atomnum - 1]

    @staticmethod
    def is_functional_group(atom, group):
        """Given a pybel atom, look up if it belongs to a function group"""
        n_atoms = [a_neighbor.GetAtomicNum() for a_neighbor in pybel.ob.OBAtomAtomIter(atom.OBAtom)]

        if group in ['quartamine', 'tertamine'] and atom.atomicnum == 7:  # Nitrogen
            # It's a nitrogen, so could be a protonated amine or quaternary ammonium
            if '1' not in n_atoms and len(n_atoms) == 4:
                return True if group == 'quartamine' else False  # It's a quat. ammonium (N with 4 residues != H)
            elif atom.OBAtom.GetHyb() == 3 and len(n_atoms) >= 3:
                return True if group == 'tertamine' else False  # It's sp3-hybridized, so could pick up an hydrogen
            else:
                return False

        if group in ['sulfonium', 'sulfonicacid', 'sulfate'] and atom.atomicnum == 16:  # Sulfur
            if '1' not in n_atoms and len(n_atoms) == 3:  # It's a sulfonium (S with 3 residues != H)
                return True if group == 'sulfonium' else False
            elif n_atoms.count(8) == 3:  # It's a sulfonate or sulfonic acid
                return True if group == 'sulfonicacid' else False
            elif n_atoms.count(8) == 4:  # It's a sulfate
                return True if group == 'sulfate' else False

        if group == 'phosphate' and atom.atomicnum == 15:  # Phosphor
            if set(n_atoms) == {8}:  # It's a phosphate
                return True

        if group in ['carboxylate', 'guanidine'] and atom.atomicnum == 6:  # It's a carbon atom
            if n_atoms.count(8) == 2 and n_atoms.count(6) == 1:  # It's a carboxylate group
                return True if group == 'carboxylate' else False
            elif n_atoms.count(7) == 3 and len(n_atoms) == 3:  # It's a guanidine group
                nitro_partners = []
                for nitro in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    nitro_partners.append(len([b_neighbor for b_neighbor in pybel.ob.OBAtomAtomIter(nitro)]))
                if min(nitro_partners) == 1:  # One nitrogen is only connected to the carbon, can pick up a H
                    return True if group == 'guanidine' else False

        if group == 'halocarbon' and atom.atomicnum in [9, 17, 35, 53]:  # Halogen atoms
            n_atoms = [na for na in pybel.ob.OBAtomAtomIter(atom.OBAtom) if na.GetAtomicNum() == 6]
            if len(n_atoms) == 1:  # Halocarbon
                return True
        else:
            return False
    
    def find_hal_acc(self, atoms):
        """Look for halogen bond acceptors (Y-{O|P|N|S}, with Y=C,P,S)"""
        data = namedtuple('hal_acceptor', 'o o_orig_idx y y_orig_idx')
        a_set = []
        # All oxygens, nitrogen, sulfurs with neighboring carbon, phosphor, nitrogen or sulfur
        for a in [at for at in atoms if at.atomicnum in [8, 7, 16]]:
            n_atoms = [na for na in pybel.ob.OBAtomAtomIter(a.OBAtom) if na.GetAtomicNum() in [6, 7, 15, 16]]
            if len(n_atoms) == 1:  # Proximal atom
                o_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
                y_orig_idx = self.Mapper.mapid(self.segment_id, n_atoms[0].GetIdx())
                a_set.append(data(o=a, o_orig_idx=o_orig_idx, y=pybel.Atom(n_atoms[0]), y_orig_idx=y_orig_idx))
        return a_set


    def find_hal_don(self, atoms):
        """Look for halogen bond donors (X-C, with X=F, Cl, Br, I)"""
        data = namedtuple('hal_donor', 'x orig_x x_orig_idx c c_orig_idx')
        a_set = []
        for a in atoms:
            if self.is_functional_group(a, 'halocarbon'):
                n_atoms = [na for na in pybel.ob.OBAtomAtomIter(a.OBAtom) if na.GetAtomicNum() == 6]
                x_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
                orig_x = self.Mapper.id_to_atom(x_orig_idx)
                c_orig_idx = [self.Mapper.mapid(self.segment_id, na.GetIdx()) for na in n_atoms]
                a_set.append(data(x=a, orig_x=orig_x, x_orig_idx=x_orig_idx,
                                  c=pybel.Atom(n_atoms[0]), c_orig_idx=c_orig_idx))
        return a_set

    def find_charged(self, all_atoms):
        """Identify all positively charged groups in a ligand. This search is not exhaustive, as the cases can be quite
        diverse. The typical cases seem to be protonated amines, quaternary ammoinium and sulfonium
        as mentioned in 'Cation-pi interactions in ligand recognition and catalysis' (Zacharias et al., 2002)).
        Identify negatively charged groups in the ligand.
        """
        data = namedtuple('lcharge', 'atoms orig_atoms atoms_orig_idx type center fgroup')
        a_set = []
        for a in all_atoms:
            if a.atomicnum == 1: # Skip hydrogens
                continue
            a_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
            a_orig = self.Mapper.id_to_atom(a_orig_idx)
            if self.is_functional_group(a, 'quartamine'):
                a_set.append(data(atoms=[a, ], orig_atoms=[a_orig, ], atoms_orig_idx=[a_orig_idx, ], type='positive',
                                  center=list(a.coords), fgroup='quartamine'))
            elif self.is_functional_group(a, 'tertamine'):
                a_set.append(data(atoms=[a, ], orig_atoms=[a_orig, ], atoms_orig_idx=[a_orig_idx, ], type='positive',
                                  center=list(a.coords),
                                  fgroup='tertamine'))
            if self.is_functional_group(a, 'sulfonium'):
                a_set.append(data(atoms=[a, ], orig_atoms=[a_orig, ], atoms_orig_idx=[a_orig_idx, ], type='positive',
                                  center=list(a.coords),
                                  fgroup='sulfonium'))
            if self.is_functional_group(a, 'phosphate'):
                a_contributing = [a, ]
                a_contributing_orig_idx = [a_orig_idx, ]
                [a_contributing.append(pybel.Atom(neighbor)) for neighbor in pybel.ob.OBAtomAtomIter(a.OBAtom)]
                [a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, neighbor.idx))
                 for neighbor in a_contributing]
                orig_contributing = [self.Mapper.id_to_atom(idx) for idx in a_contributing_orig_idx]
                a_set.append(
                    data(atoms=a_contributing, orig_atoms=orig_contributing, atoms_orig_idx=a_contributing_orig_idx,
                         type='negative',
                         center=a.coords, fgroup='phosphate'))
            if self.is_functional_group(a, 'sulfonicacid'):
                a_contributing = [a, ]
                a_contributing_orig_idx = [a_orig_idx, ]
                [a_contributing.append(pybel.Atom(neighbor)) for neighbor in pybel.ob.OBAtomAtomIter(a.OBAtom) if
                 neighbor.GetAtomicNum() == 8]
                [a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, neighbor.idx))
                 for neighbor in a_contributing]
                orig_contributing = [self.Mapper.id_to_atom(idx) for idx in a_contributing_orig_idx]
                a_set.append(
                    data(atoms=a_contributing, orig_atoms=orig_contributing, atoms_orig_idx=a_contributing_orig_idx,
                         type='negative',
                         center=a.coords, fgroup='sulfonicacid'))
            elif self.is_functional_group(a, 'sulfate'):
                a_contributing = [a, ]
                a_contributing_orig_idx = [a_orig_idx, ]
                [a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, neighbor.idx))
                 for neighbor in a_contributing]
                [a_contributing.append(pybel.Atom(neighbor)) for neighbor in pybel.ob.OBAtomAtomIter(a.OBAtom)]
                orig_contributing = [self.Mapper.id_to_atom(idx) for idx in a_contributing_orig_idx]
                a_set.append(
                    data(atoms=a_contributing, orig_atoms=orig_contributing, atoms_orig_idx=a_contributing_orig_idx,
                         type='negative',
                         center=a.coords, fgroup='sulfate'))
            if self.is_functional_group(a, 'carboxylate'):
                a_contributing = [pybel.Atom(neighbor) for neighbor in pybel.ob.OBAtomAtomIter(a.OBAtom)
                                  if neighbor.GetAtomicNum() == 8]
                a_contributing_orig_idx = [self.Mapper.mapid(self.segment_id, neighbor.idx)
                                           for neighbor in a_contributing]
                orig_contributing = [self.Mapper.id_to_atom(idx) for idx in a_contributing_orig_idx]
                a_set.append(
                    data(atoms=a_contributing, orig_atoms=orig_contributing, atoms_orig_idx=a_contributing_orig_idx,
                         type='negative',
                         center=centroid([a.coords for a in a_contributing]), fgroup='carboxylate'))
            elif self.is_functional_group(a, 'guanidine'):
                a_contributing = [pybel.Atom(neighbor) for neighbor in pybel.ob.OBAtomAtomIter(a.OBAtom)
                                  if neighbor.GetAtomicNum() == 7]
                a_contributing_orig_idx = [self.Mapper.mapid(self.segment_id, neighbor.idx)
                                           for neighbor in a_contributing]
                orig_contributing = [self.Mapper.id_to_atom(idx) for idx in a_contributing_orig_idx]
                a_set.append(
                    data(atoms=a_contributing, orig_atoms=orig_contributing, atoms_orig_idx=a_contributing_orig_idx,
                         type='positive',
                         center=a.coords, fgroup='guanidine'))
        return a_set

    def find_metal_binding(self, lig_atoms): #, water_oxygens):
        """Looks for atoms that could possibly be involved in binding a metal ion.
        This can be any water oxygen, as well as oxygen from carboxylate, phophoryl, phenolate, alcohol;
        nitrogen from imidazole; sulfur from thiolate.
        """
        hetid = 'hetid'
        position = -1
        chain = 'chain'

        a_set = []
        data = namedtuple('metal_binding', 'atom orig_atom atom_orig_idx type fgroup restype resnr reschain location')
        # for oxygen in water_oxygens:
        #     a_set.append(data(atom=oxygen.oxy, atom_orig_idx=oxygen.oxy_orig_idx, type='O', fgroup='water',
        #                       restype=whichrestype(oxygen.oxy), resnr=whichresnumber(oxygen.oxy),
        #                       reschain=whichchain(oxygen.oxy), location='water',
        #                       orig_atom=self.Mapper.id_to_atom(oxygen.oxy_orig_idx)))
        # #@todo Refactor code
        for a in lig_atoms:
            if a.atomicnum == 1: # Skip hydrogens
                continue
            a_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
            n_atoms = pybel.ob.OBAtomAtomIter(a.OBAtom)  # Neighboring atoms
            # All atomic numbers of neighboring atoms
            n_atoms_atomicnum = [n.GetAtomicNum() for n in pybel.ob.OBAtomAtomIter(a.OBAtom)]
            if a.atomicnum == 8:  # Oxygen
                if n_atoms_atomicnum.count('1') == 1 and len(n_atoms_atomicnum) == 2:  # Oxygen in alcohol (R-[O]-H)
                    a_set.append(data(atom=a, atom_orig_idx=a_orig_idx, type='O', fgroup='alcohol',
                                      location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx),
                                      restype=hetid, resnr=position, reschain=chain))
                if True in [n.IsAromatic() for n in n_atoms] and not a.OBAtom.IsAromatic():  # Phenolate oxygen
                    a_set.append(data(atom=a, atom_orig_idx=a_orig_idx, type='O', fgroup='phenolate',
                                      location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx),
                                      restype=hetid, resnr=position, reschain=chain))
            if a.atomicnum == 6:  # It's a carbon atom
                if n_atoms_atomicnum.count(8) == 2 and n_atoms_atomicnum.count(6) == 1:  # It's a carboxylate group
                    for neighbor in [n for n in n_atoms if n.GetAtomicNum() == 8]:
                        neighbor_orig_idx = self.Mapper.mapid(self.segment_id, neighbor.GetIdx())
                        a_set.append(data(atom=pybel.Atom(neighbor), atom_orig_idx=neighbor_orig_idx, type='O',
                                          fgroup='carboxylate', restype=hetid, resnr=position, reschain=chain,
                                          location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx)))
            if a.atomicnum == 15:  # It's a phosphor atom
                if n_atoms_atomicnum.count(8) >= 3:  # It's a phosphoryl
                    for neighbor in [n for n in n_atoms if n.GetAtomicNum() == 8]:
                        neighbor_orig_idx = self.Mapper.mapid(self.segment_id, neighbor.GetIdx())
                        a_set.append(data(atom=pybel.Atom(neighbor), atom_orig_idx=neighbor_orig_idx, type='O',
                                          fgroup='phosphoryl', restype=hetid, resnr=position, reschain=chain,
                                          location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx)))
                if n_atoms_atomicnum.count(8) == 2:  # It's another phosphor-containing group #@todo (correct name?)
                    for neighbor in [n for n in n_atoms if n.GetAtomicNum() == 8]:
                        neighbor_orig_idx = self.Mapper.mapid(self.segment_id, neighbor.GetIdx())
                        a_set.append(data(atom=pybel.Atom(neighbor), atom_orig_idx=neighbor_orig_idx, type='O',
                                          fgroup='phosphor.other', restype=hetid, resnr=position, reschain=chain,
                                          location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx)))
            if a.atomicnum == 7:  # It's a nitrogen atom
                if n_atoms_atomicnum.count(6) == 2:  # It's imidazole/pyrrole or similar
                    a_set.append(data(atom=a, atom_orig_idx=a_orig_idx, type='N', fgroup='imidazole/pyrrole',
                                      location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx),
                                      restype=hetid, resnr=position, reschain=chain))
            if a.atomicnum == 16:  # It's a sulfur atom
                if True in [n.IsAromatic() for n in n_atoms] and not a.OBAtom.IsAromatic():  # Thiolate
                    a_set.append(data(atom=a, atom_orig_idx=a_orig_idx, type='S', fgroup='thiolate',
                                      location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx),
                                      restype=hetid, resnr=position, reschain=chain))
                if set(n_atoms_atomicnum) == {26}:  # Sulfur in Iron sulfur cluster
                    a_set.append(data(atom=a, atom_orig_idx=a_orig_idx, type='S', fgroup='iron-sulfur.cluster',
                                      location=self.segment_id, orig_atom=self.Mapper.id_to_atom(a_orig_idx),
                                      restype=hetid, resnr=position, reschain=chain))

        return a_set

class BindingSite(Mol):
    def __init__(self, mapper, segment_id):
        """Find all relevant parts which could take part in interactions"""
        Mol.__init__(self, mapper, segment_id)
        self.all_atoms = self.original_mol.atoms
        self.rings = self.find_rings(self.original_mol, self.all_atoms)
        self.hydroph_atoms = self.hydrophobic_atoms(self.all_atoms)
        self.hbond_acc_atoms = self.find_hba(self.all_atoms)
        self.hbond_don_atom_pairs = self.find_hbd(self.all_atoms, self.hydroph_atoms)
        self.charged = self.find_charged(self.original_mol)
        self.halogenbond_acc = self.find_hal(self.all_atoms)
        self.metal_binding = self.find_metal_binding(self.original_mol)

    def find_hal(self, atoms):
        """Look for halogen bond acceptors (Y-{O|P|N|S}, with Y=C,P,S)"""
        data = namedtuple('hal_acceptor', 'o o_orig_idx y y_orig_idx')
        a_set = []
        # All oxygens, nitrogen, sulfurs with neighboring carbon, phosphor, nitrogen or sulfur
        for a in [at for at in atoms if at.atomicnum in [8, 7, 16]]:
            n_atoms = [na for na in pybel.ob.OBAtomAtomIter(a.OBAtom) if na.GetAtomicNum() in [6, 7, 15, 16]]
            if len(n_atoms) == 1:  # Proximal atom
                o_orig_idx = self.Mapper.mapid(self.segment_id, a.idx)
                y_orig_idx = self.Mapper.mapid(self.segment_id, n_atoms[0].GetIdx())
                a_set.append(data(o=a, o_orig_idx=o_orig_idx, y=pybel.Atom(n_atoms[0]), y_orig_idx=y_orig_idx))
        return a_set

    def find_charged(self, mol):
        """Looks for positive charges in arginine, histidine or lysine, for negative in aspartic and glutamic acid."""
        """If nucleic acids are part of the receptor, looks for negative charges in phosphate backbone"""
        data = namedtuple('pcharge', 'atoms atoms_orig_idx type center restype resnr reschain')
        a_set = []
        # Iterate through all residue, exclude those in chains defined as peptides
        for res in [r for r in pybel.ob.OBResidueIter(mol.OBMol) if not r.GetChain() in config.PEPTIDES]:
            if config.INTRA is not None:
                if res.GetChain() != config.INTRA:
                    continue
            a_contributing = []
            a_contributing_orig_idx = []
            if res.GetName() in ('ARG', 'HIS', 'LYS'):  # Arginine, Histidine or Lysine have charged sidechains
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('N') and res.GetAtomProperty(a, 8):
                        a_contributing.append(pybel.Atom(a))
                        a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, a.GetIdx()))
                if not len(a_contributing) == 0:
                    a_set.append(data(atoms=a_contributing,
                                      atoms_orig_idx=a_contributing_orig_idx,
                                      type='positive',
                                      center=centroid([ac.coords for ac in a_contributing]),
                                      restype=res.GetName(),
                                      resnr=res.GetNum(),
                                      reschain=res.GetChain()))
            if res.GetName() in ('GLU', 'ASP'):  # Aspartic or Glutamic Acid
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('O') and res.GetAtomProperty(a, 8):
                        a_contributing.append(pybel.Atom(a))
                        a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, a.GetIdx()))
                if not len(a_contributing) == 0:
                    a_set.append(data(atoms=a_contributing,
                                      atoms_orig_idx=a_contributing_orig_idx,
                                      type='negative',
                                      center=centroid([ac.coords for ac in a_contributing]),
                                      restype=res.GetName(),
                                      resnr=res.GetNum(),
                                      reschain=res.GetChain()))
            if res.GetName() in config.DNA + config.RNA: # and config.DNARECEPTOR: # nucleic acids have negative charge in sugar phosphate
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('P') and res.GetAtomProperty(a, 9):
                        a_contributing.append(pybel.Atom(a))
                        a_contributing_orig_idx.append(self.Mapper.mapid(self.segment_id, a.GetIdx()))
                if not len(a_contributing) == 0:
                    a_set.append(data(atoms=a_contributing,atoms_orig_idx=a_contributing_orig_idx, type='negative', 
                                      center=centroid([ac.coords for ac in a_contributing]), restype=res.GetName(),
                                      resnr=res.GetNum(),
                                      reschain=res.GetChain()))
        return a_set

    def find_metal_binding(self, mol):
        """Looks for atoms that could possibly be involved in chelating a metal ion.
        This can be any main chain oxygen atom or oxygen, nitrogen and sulfur from specific amino acids"""
        data = namedtuple('metal_binding', 'atom atom_orig_idx type restype resnr reschain location')
        a_set = []
        for res in pybel.ob.OBResidueIter(mol.OBMol):
            restype, reschain, resnr = res.GetName().upper(), res.GetChain(), res.GetNum()
            if restype in ['ASP', 'GLU', 'SER', 'THR', 'TYR']:  # Look for oxygens here
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('O') and res.GetAtomProperty(a, 8):
                        atom_orig_idx = self.Mapper.mapid(self.segment_id, a.GetIdx())
                        a_set.append(data(atom=pybel.Atom(a), atom_orig_idx=atom_orig_idx, type='O', restype=restype,
                                          resnr=resnr, reschain=reschain,
                                          location='protein.sidechain'))
            if restype == 'HIS':  # Look for nitrogen here
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('N') and res.GetAtomProperty(a, 8):
                        atom_orig_idx = self.Mapper.mapid(self.segment_id, a.GetIdx())
                        a_set.append(data(atom=pybel.Atom(a), atom_orig_idx=atom_orig_idx, type='N', restype=restype,
                                          resnr=resnr, reschain=reschain,
                                          location='protein.sidechain'))
            if restype == 'CYS':  # Look for sulfur here
                for a in pybel.ob.OBResidueAtomIter(res):
                    if a.GetType().startswith('S') and res.GetAtomProperty(a, 8):
                        atom_orig_idx = self.Mapper.mapid(self.segment_id, a.GetIdx())
                        a_set.append(data(atom=pybel.Atom(a), atom_orig_idx=atom_orig_idx, type='S', restype=restype,
                                          resnr=resnr, reschain=reschain,
                                          location='protein.sidechain'))
            for a in pybel.ob.OBResidueAtomIter(res):  # All main chain oxygens
                if a.GetType().startswith('O') and res.GetAtomProperty(a, 2) and restype != 'HOH':
                    atom_orig_idx = self.Mapper.mapid(self.segment_id, a.GetIdx())
                    a_set.append(data(atom=pybel.Atom(a), atom_orig_idx=atom_orig_idx, type='O', restype=res.GetName(),
                                      resnr=res.GetNum(), reschain=res.GetChain(),
                                      location='protein.mainchain'))
        return a_set


class DataMapper:
    """Provides functions for mapping atom IDs in the correct way"""

    def __init__(self, data, pybel_mols):
        self.data = data # raw data
        self.pybel_mols = pybel_mols # openbabel molecules of the data
        self.idx_map = {} # map pybel indexes to original indexes
        self.original_map = {} # map original indexes to pybel indexes
        self.block_ids = [id for id, length in enumerate(data["block_lengths"]) for _ in range(length)]
        
        segment_start = 0
        block_segment_ids = np.array(data["segment_ids"])
        block_len = np.array(data["block_lengths"])
        for segment_id in sorted(set(data['segment_ids'])):
            idx_map = {}
            segment_len = block_len[block_segment_ids == segment_id].sum()
            segment_end = segment_start + segment_len
            if VOCAB.idx_to_atom(data['A'][segment_start]) == VOCAB.atom_global:
                self.original_map[segment_start] = None # global node is not a pybel atom
                segment_start += 1
            for i in range(segment_start, segment_end):
                # atoms = [a for a in pybel_mols[segment_id].atoms if a.atomicnum != 1] # hydrogens are not in the data
                atoms = pybel_mols[segment_id].atoms
                # assert len(atoms) == segment_end - segment_start
                atom_idx = atoms[i - segment_start].idx
                idx_map[atom_idx] = i
                self.original_map[i] = (segment_id, i - segment_start)
            self.idx_map[segment_id] = idx_map
            segment_start = segment_end
    
    def mapid(self, segment_id, pybabel_idx):  # Mapping to original IDs
        return self.idx_map[segment_id][pybabel_idx]

    def id_to_atom(self, original_idx):
        """Returns the atom for a given original ligand ID.
        To do this, the ID is mapped to the protein first and then the atom returned.
        """
        res = self.original_map[original_idx]
        if res is None:
            return None
        segment_id, mapped_idx = res
        return self.pybel_mols[segment_id].atoms[mapped_idx]

    def id_to_block(self, original_idx):
        """Returns the block ID for a given original ligand ID."""
        return self.block_ids[original_idx]


def saltbridge(poscenter, negcenter, protispos, noprot):
    """Detect all salt bridges (pliprofiler between centers of positive and negative charge)"""
    data = namedtuple(
        'saltbridge', 'positive negative distance protispos resnr restype reschain resnr_l restype_l reschain_l')
    pairings = []
    for pc, nc in itertools.product(poscenter, negcenter):
        if not config.MIN_DIST < euclidean3d(pc.center, nc.center) < config.SALTBRIDGE_DIST_MAX:
            continue
        if noprot:
            resnr = whichresnumber(pc.orig_atoms[0]) if protispos else whichresnumber(nc.orig_atoms[0])
            restype = whichrestype(pc.orig_atoms[0]) if protispos else whichrestype(nc.orig_atoms[0])
            reschain = whichchain(pc.orig_atoms[0]) if protispos else whichchain(nc.orig_atoms[0])
        else:
            resnr = pc.resnr if protispos else nc.resnr
            restype = pc.restype if protispos else nc.restype
            reschain = pc.reschain if protispos else nc.reschain
        resnr_l = whichresnumber(nc.orig_atoms[0]) if protispos else whichresnumber(pc.orig_atoms[0])
        restype_l = whichrestype(nc.orig_atoms[0]) if protispos else whichrestype(pc.orig_atoms[0])
        reschain_l = whichchain(nc.orig_atoms[0]) if protispos else whichchain(pc.orig_atoms[0])
        contact = data(positive=pc, negative=nc, distance=euclidean3d(pc.center, nc.center), protispos=protispos,
                       resnr=resnr, restype=restype, reschain=reschain, resnr_l=resnr_l, restype_l=restype_l,
                       reschain_l=reschain_l)
        pairings.append(contact)
    return filter_contacts(pairings)

class InteractionProfile:
    def __init__(self, item, pybel_mols, *segment_types: Tuple[SegmentType]):
        self.item = item
        self.data = data = item['data']
        self.pybel_mols = pybel_mols
        self.segment_types = segment_types
        assert len(segment_types) == len(pybel_mols) == len(set(data["segment_ids"])), f"{len(segment_types)}, {len(pybel_mols)}, {len(set(data['segment_ids']))}"

        self.mapper = DataMapper(data, pybel_mols)

        self.molecules = []
        for segment_id, segment_type in enumerate(segment_types):
            if segment_type == SegmentType.LIGAND:
                self.molecules.append(Ligand(self.mapper, segment_id))
            elif segment_type == SegmentType.BINDING_SITE:
                self.molecules.append(BindingSite(self.mapper, segment_id))
            else:
                raise ValueError("Invalid segment type")

        if len(self.molecules) > 2:
            raise NotImplementedError("Only two molecules are supported by InteractionProfile")
        
        self.mol0 = self.molecules[0]
        self.mol1 = self.molecules[1]

        noprot = all([segment_type == SegmentType.LIGAND for segment_type in segment_types])
        self.saltbridge_lneg = saltbridge(self.mol0.get_pos_charged(), self.mol1.get_neg_charged(), True, noprot)
        self.saltbridge_pneg = saltbridge(self.mol1.get_pos_charged(), self.mol0.get_neg_charged(), False, noprot)

        self.all_hbonds_ldon = hbonds(self.mol0.get_hba(), self.mol1.get_hbd(), False, 'strong')
        self.all_hbonds_pdon = hbonds(self.mol1.get_hba(), self.mol0.get_hbd(), True, 'strong')

        self.hbonds_ldon = PLInteraction.refine_hbonds_ldon(self.all_hbonds_ldon, self.saltbridge_lneg, self.saltbridge_pneg)
        self.hbonds_pdon = PLInteraction.refine_hbonds_pdon(self.all_hbonds_pdon, self.saltbridge_lneg, self.saltbridge_pneg)

        self.pistacking = pistacking(self.mol0.rings, self.mol1.rings)

        self.all_pi_cation_laro = pication(self.mol1.rings, self.mol0.get_pos_charged(), True)
        self.pication_paro = pication(self.mol0.rings, self.mol1.get_pos_charged(), False)



        self.pication_laro = PLInteraction.refine_pi_cation_laro(self.all_pi_cation_laro, self.pistacking)

        self.all_hydrophobic_contacts = hydrophobic_interactions(self.mol0.get_hydrophobic_atoms(), self.mol1.get_hydrophobic_atoms())
        self.hydrophobic_contacts = PLInteraction.refine_hydrophobic(self.all_hydrophobic_contacts, self.pistacking)

        if type(self.mol1) == Ligand:
            self.halogen_bonds_ldon = halogen(self.mol0.halogenbond_acc, self.mol1.halogenbond_don)
        else:
            self.halogen_bonds_ldon = []

        if type(self.mol0) == Ligand:
            self.halogen_bonds_pdon = halogen(self.mol1.halogenbond_acc, self.mol0.halogenbond_don)
        else:
            self.halogen_bonds_pdon = []
        
        if type(self.mol1) == Ligand:
            self.metal_complexes = metal_complexation(self.mol1.metals, self.mol1.metal_binding, self.mol0.metal_binding)
        else:
            self.metal_complexes = []


    def __str__(self):
        return f"""
        Num Saltbridge Lneg: {len(self.saltbridge_lneg)}, Num Saltbridge Pneg: {len(self.saltbridge_pneg)}, HBonds Ldon: {len(self.hbonds_ldon)}, HBonds Pdon: {len(self.hbonds_pdon)},
        Num Pi-stacking: {len(self.pistacking)}, Num Hydrophobic contacts: {len(self.hydrophobic_contacts)}, Num Pication Laro: {len(self.pication_laro)}, Num Pication Paro: {len(self.pication_paro)}, 
        Num Halogen bonds Ldon: {len(self.halogen_bonds_ldon)}, Num Halogen bonds Pdon: {len(self.halogen_bonds_pdon)}, Num Metal Complexes: {len(self.metal_complexes)}
        """
    
    @staticmethod
    def _convert_data(out_dict):
        # Convert Pybel Atoms to their indexes
        for key, value in out_dict.items():
            if type(value) == openbabel.openbabel.OBRing:
                out_dict[key] = value.ring_id
            if type(value) == np.ndarray:
                value = value.tolist()
            if isinstance(value, tuple) and hasattr(value, '_asdict'):
                out_dict[key] = InteractionProfile._convert_data(value._asdict())
            if isinstance(value, dict):
                out_dict[key] = InteractionProfile._convert_data(value)
            if isinstance(value, list):
                new_value = []
                for v in value:
                    if isinstance(v, pybel.Atom):
                        new_value.append({
                            "atom_idx": v.idx,
                            "atomicnum": v.atomicnum, 
                            "coords": v.coords})
                    else:
                        new_value.append(v)
                out_dict[key] = new_value
            if isinstance(value, pybel.Atom):
                out_dict[key] = {
                    "atom_idx": value.idx,
                    "atomicnum": value.atomicnum, 
                    "coords": value.coords
                }
        return out_dict
    
    @property
    def summary(self):
        output = {
            "id": self.item["id"],
            "saltbridge_lneg": self.saltbridge_lneg,
            "saltbridge_pneg": self.saltbridge_pneg,
            "hbonds_ldon": self.hbonds_ldon,
            "hbonds_pdon": self.hbonds_pdon,
            "pistacking": self.pistacking,
            "hydrophobic_contacts": self.hydrophobic_contacts,
            "pication_laro": self.pication_laro,
            "pication_paro": self.pication_paro,
            "halogen_bonds_ldon": self.halogen_bonds_ldon,
            "halogen_bonds_pdon": self.halogen_bonds_pdon,
            "metal_complexes": self.metal_complexes
        }
        for key, value in output.items():
            if key == "id":
                continue
            output[key] = [self._convert_data(contact._asdict()) for contact in value]
        return output