from .mol_atom_match import struct_to_topology
import networkx as nx
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import torch

# Adapted from https://github.com/gcorso/DiffDock/blob/main/utils/torsion.py

def get_torsion_mask(atoms, coords):
    """
    Gets the torsion mask for the atoms and coordinates.
    A bond can have torsion around it if breaking it would disconnect the molecule.
    i.e. rings do not have torsion 
    Note: this does not consider if the bond is double or triple    
    Args:
        atoms: [n_atoms], list of atom types
        coords: [n_atoms, 3], list of atom coordinates
    Returns:
        edges: [n_edges, 2]
        mask_edges: [n_edges], True if the edge is rotatable
        mask_rotate: [n_rotatable_edges, n_atoms], True if the atom is part of the 
         part that gets rotated, n_rotatable_edges = sum(mask_edges)
    """
    G = struct_to_topology(atoms, coords) # gets bonds
    to_rotate = []
    edges = list(nx.edges(G))
    mask_edges, mask_rotate = [], []
    for i in range(0, len(edges)):
        G2 = G.copy()
        num_connected_components = nx.number_connected_components(G2)
        G2.remove_edge(*edges[i])
        if nx.number_connected_components(G2) > num_connected_components:
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


def get_backbone_mask(atoms, atom_positions, is_nucleotide: bool = False):
    if is_nucleotide:
        backbone_mask = []
        backbone_items = ["'", "P"] # P, O1P, O2P, O5', C5', C4', C3', O3', C2', C1'
        for atom, atom_pos in zip(atoms, atom_positions):
            if atom == "P":
                backbone_mask.append(True)
            elif atom_pos in backbone_items:
                backbone_mask.append(True)
            else:
                backbone_mask.append(False)
        return np.array(backbone_mask)
    else: # amino acids
        backbone_items = ["A", ""] # N, CA, C, O
        return np.array([atom_pos in backbone_items for atom_pos in atom_positions])


def get_segment_torsion_mask(blocks):
    atoms = [unit.element for block in blocks for unit in block.units]
    coords = [unit.coordinate for block in blocks for unit in block.units]
    return get_torsion_mask(atoms, coords)


def get_side_chain_torsion_mask(blocks):
    """
    Gets the side chain torsion mask for each block.
    Args:
        blocks: [n_blocks], list of blocks
    Returns:
        edges: [n_edges, 2], list of edges
        sidechain_mask_edges: [n_edges], True if the edge is rotatable
        sidechain_mask_rotate: [n_rotatable_edges, n_atoms], True if the atom is 
         part of the part that gets rotated, n_rotatable_edges = sum(mask_edges)
    """
    all_edges, all_sidechain_mask_edges, all_sidechain_mask_rotate = [], [], []
    total_atoms = len([unit for block in blocks for unit in block.units])
    curr_atom = 0
    for block in blocks:
        atoms = [unit.element for unit in block.units]
        coords = [unit.coordinate for unit in block.units]
        atom_pos = [unit.pos_code for unit in block.units]
        is_nucleotide = block.symbol in {"DA", "DT", "DC", "DG", "<G>", "RU", "RA", "RG", "RC", "RI"}
        backbone_atom_mask = get_backbone_mask(atoms, atom_pos, is_nucleotide=is_nucleotide)
        edges, mask_edges, mask_rotate = get_torsion_mask(atoms, coords)

        sidechain_mask_rotate = []
        sidechain_mask_edges = mask_edges.copy()
        mask_rotate_idx = 0
        for idx, (u, v) in enumerate(edges):
            if mask_edges[idx] == False:
                continue
            if backbone_atom_mask[u] and backbone_atom_mask[v]:
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
                    assert not np.any(backbone_atoms), f"Error: make sure that no backbone atoms are being rotated"
                    sidechain_mask_rotate.append(mask_rotate_)
                mask_rotate_idx += 1
            
        all_edges.append(edges+curr_atom)
        all_sidechain_mask_edges.append(sidechain_mask_edges)
        start_pad = curr_atom
        end_pad = total_atoms - len(atoms) - start_pad
        for mask in sidechain_mask_rotate:
            padded_mask = np.concatenate([np.zeros(start_pad, dtype=bool), mask, np.zeros(end_pad, dtype=bool)])
            all_sidechain_mask_rotate.append(padded_mask)
        curr_atom += len(atoms)
    return np.concatenate(all_edges, axis=0), np.concatenate(all_sidechain_mask_edges), np.array(all_sidechain_mask_rotate)
    