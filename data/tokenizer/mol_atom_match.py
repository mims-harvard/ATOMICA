#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
from itertools import combinations
from math import sqrt
from typing import List, Tuple, Dict, Union

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import Mol as RDKitMol


# copied from https://github.com/zotko/xyz2graph/blob/master/xyz2graph/xyz2graph.py
atomic_radii = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.23,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
)


def _mol_to_topology(mol: Union[RDKitMol, str], include_Hs: bool=False):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        Chem.Kekulize(mol, True)
    g = nx.Graph()
    in_graph = []

    # add nodes
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        if symbol != 'H' or include_Hs:
            g.add_node(i, atom=symbol)
            in_graph.append(True)
        else:
            in_graph.append(False)

    # add edges
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if in_graph[i] and in_graph[j]:
            g.add_edge(i, j, bond_type=bond.GetBondType())

    return g
    

def get_atom_map(
        g1: Union[nx.Graph, RDKitMol, str],
        g2: Union[nx.Graph, RDKitMol, str]
    ) -> Dict: # mapping from g1 nodes to g2 nodes
    if not isinstance(g1, nx.Graph):
        g1 = _mol_to_topology(g1)
    if not isinstance(g2, nx.Graph):
        g2 = _mol_to_topology(g2)
    gm = isomorphism.GraphMatcher(g1, g2, node_match=lambda n1, n2: n1['atom'] == n2['atom'])
    assert gm.is_isomorphic(), f'g1 node {len(g1)}, g2 node {len(g2)}'
    return gm.mapping


def struct_to_topology(
        atoms: List[str],
        coordinates: List[Tuple[float, float, float]]
    ) -> nx.Graph:
    node_ids = list(range(len(atoms)))
    coordinates = np.array(coordinates) # [N, 3]
    dist = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]  # [N, N, 3]
    dist = np.linalg.norm(dist, axis=-1) # [N, N]

    radius = np.array([atomic_radii[atom] for atom in atoms])  # [N]
    dist_bond = (radius[:, np.newaxis] + radius[np.newaxis, :]) * 1.3  # [N, N]

    adj_mat = np.logical_and(0.1 < dist, dist_bond > dist)

    g = nx.Graph()

    for i in node_ids:
        g.add_node(i, atom=atoms[i])

    for i, j in zip(*np.nonzero(adj_mat)):
        g.add_edge(int(i), int(j))

    return g


def struct_to_bonds(
        atoms: List[str],
        coordinates: List[Tuple[float, float, float]],
        smiles: str,
        include_Hs: bool=False
    ) -> Tuple[int, int, int]:
    bond2id = {
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4
    }

    g1 = _mol_to_topology(smiles, include_Hs)
    g2 = struct_to_topology(atoms, coordinates)
    matching = get_atom_map(g1, g2)
    bonds = []
    for edge in g1.edges.data():
        i, j, attr = edge
        bonds.append((
            matching[i],
            matching[j],
            bond2id[attr['bond_type']]
        ))

    return bonds


if __name__ == '__main__':
    # benzene
    order = [2, 4, 1, 0, 5, 3]
    g = nx.Graph()
    for i in order:
        g.add_node(i, atom='C')

    for i in range(6):
        j = i + 1
        if j == len(order):
            j = 0
        g.add_edge(order[i], order[j])

    print(get_atom_map(g, 'c1ccccc1'))