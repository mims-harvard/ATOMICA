from .mol_atom_match import struct_to_topology
import networkx as nx
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import torch

# Adapted from https://github.com/gcorso/DiffDock/blob/main/utils/torsion.py

def get_torsion_mask(atoms, coords):
    # TODO: if tokenization is used, add in a filter for double and triple bonds
    G = struct_to_topology(atoms, coords) # gets bonds
    to_rotate = []
    edges = list(nx.edges(G))
    mask_edges, mask_rotate = [], []
    for i in range(0, len(edges)):
        G2 = G.copy()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            assert nx.number_connected_components(G2) == 2
            l1 = list(sorted(nx.connected_components(G2), key=len)[0])
            l2 = list(sorted(nx.connected_components(G2), key=len)[1])
            if len(l1) > 1 and len(l2) > 1:
                to_rotate = []
                for i in range(len(G.nodes())):
                    if i in l1:
                        to_rotate.append(True)
                    else:
                        to_rotate.append(False)
                mask_rotate.append(to_rotate)
                mask_edges.append(True)
            else:
                mask_edges.append(False)
        else:
            mask_edges.append(False)
    return np.array(edges), np.array(mask_edges), np.array(mask_rotate)


def modify_conformer_torsion_angles(coords, rotateable_edges, mask_rotate, torsion_updates, as_numpy=False):
    coords = copy.deepcopy(coords)
    if type(coords) == torch.Tensor: 
        coords = coords.cpu().numpy()
    elif type(coords) == list:
        coords = np.array(coords)

    for idx_edge, e in enumerate(rotateable_edges):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        if int(mask_rotate[idx_edge, u]) == 0 and int(mask_rotate[idx_edge, v]) == 1:
            rot_vec = coords[u] - coords[v]  # convention: positive rotation if pointing inwards
        elif int(mask_rotate[idx_edge, u]) == 1 and int(mask_rotate[idx_edge, v]) == 0:
            rot_vec = coords[v] - coords[u]
        else:
            raise ValueError(f"Invalid edge {e} for rotation, check mask rotate.")
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        
        # mask_rotate[idx_edge][node_idx]=True, node is part of the part that gets rotated
        coords[mask_rotate[idx_edge]] = (coords[mask_rotate[idx_edge]] - coords[v]) @ rot_mat.T + coords[v]

    # if not as_numpy: coords = torch.from_numpy(coords.astype(np.float32))
    return coords


def get_side_chain_torsion_mask(block):
    # Rotate side chain atoms in amino acid block
    atoms = [unit.element for unit in block.units]
    coords = [unit.coordinate for unit in block.units]
    atom_pos = [unit.pos_code for unit in block.units]
    backbone_atom_mask = np.array([atom_pos in ["A", ""] for atom_pos in atom_pos])
    edges, mask_edges, mask_rotate = get_torsion_mask(atoms, coords)

    sidechain_mask_rotate = []
    sidechain_mask_edges = mask_edges.copy()
    mask_rotate_idx = 0
    for idx, (u, v) in enumerate(edges):
        if mask_edges[idx] == False:
            continue
        if atom_pos[u] in ["A", ""] and atom_pos[v] in ["A", ""]:
            # if both are backbone atoms, then we don't want to rotate
            sidechain_mask_edges[idx] = False
            mask_rotate_idx += 1
        else:
            # (side chain atom, side chain atom) or (side chain atom, backbone atom)
            # make sure that the rotated atoms are not in the backbone
            mask_rotate_ = mask_rotate[mask_rotate_idx]
            backbone_atoms = mask_rotate_[backbone_atom_mask] # False = backbone atom, True = side chain atom
            if not np.any(backbone_atoms):
                sidechain_mask_rotate.append(mask_rotate_)
            else:
                mask_rotate_ = np.bitwise_not(mask_rotate_)
                backbone_atoms = mask_rotate_[backbone_atom_mask] # False = backbone atom, True = side chain atom
                assert not np.any(backbone_atoms), "Error: make sure that no backbone atoms are being rotated"
                sidechain_mask_rotate.append(mask_rotate_)
            mask_rotate_idx += 1
    return edges, np.array(sidechain_mask_edges), np.array(sidechain_mask_rotate)