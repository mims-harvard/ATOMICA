import torch
import numpy as np
import copy
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
import math
from torch.nn import functional as F
from .torus import score as torus_score
from data.dataset import data_to_blocks, blocks_to_data, VOCAB
from data.torsion import get_side_chain_torsion_mask_block, get_segment_torsion_mask


def modify_conformer_torsion_angles(coords, rotateable_edges, mask_rotate, torsion_updates):
    coords = copy.deepcopy(coords)
    if type(coords) == torch.Tensor: 
        coords = coords.cpu().numpy()
    elif type(coords) == list:
        coords = np.array(coords)
    
    if type(mask_rotate) == list:
        mask_rotate = np.array(mask_rotate)

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
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        # mask_rotate[idx_edge][node_idx]=True, node is part of the part that gets rotated
        coords[mask_rotate[idx_edge]] = (coords[mask_rotate[idx_edge]] - coords[v]) @ rot_mat.T + coords[v]
    return coords


def rigid_transform_Kabsch_3D(A, B):
    # Source: https://github.com/HannesStark/EquiBind/blob/main/commons/geometry_utils.py
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

class CropTransform:
    def __init__(self, max_blocks, fragmentation_method):
        self.max_blocks = max_blocks
        self.fragmentation_method = fragmentation_method
        self.top_k = 5 # closest blocks to crop around
        self.residues = set([x[0] for x in VOCAB.aas] + [x[0] for x in VOCAB.bases])

    def __call__(self, data):
        segment0, segment1 = data_to_blocks(data, self.fragmentation_method)
        segment0_coords = np.array([block.coords for block in segment0])
        segment1_coords = np.array([block.coords for block in segment1])
        cross_segment_distances = distance.cdist(segment0_coords, segment1_coords, 'euclidean')
        top_k_indices = np.argsort(cross_segment_distances, axis=None)[:self.top_k]
        # pick one of the closest k pairs of blocks as the center
        segment0_center_index, segment1_center_index = np.unravel_index(np.random.choice(top_k_indices), cross_segment_distances.shape) 
        # crop the segment to the center
        segment0_distances = np.linalg.norm(segment0_coords - segment0_coords[segment0_center_index], axis=1)
        segment1_distances = np.linalg.norm(segment1_coords - segment1_coords[segment1_center_index], axis=1)
        keep_indices = np.argsort(np.concatenate([segment0_distances, segment1_distances]))[:self.max_blocks]
        segment0_keep_indices = keep_indices[keep_indices < len(segment0)]
        segment1_keep_indices = keep_indices[keep_indices >= len(segment0)] - len(segment0)
        segment0_cropped = [segment0[i] for i in segment0_keep_indices]
        segment1_cropped = [segment1[i] for i in segment1_keep_indices]
        cropped_data = blocks_to_data(segment0_cropped, segment1_cropped)

        kept_old_indices = np.concatenate([segment0_keep_indices+1, segment1_keep_indices+len(segment0)+2]).tolist()

        # add back torsion mask
        if 'torsion_mask' in data:
            list_of_masks = []
            for blocks in [segment0_cropped, segment1_cropped]:
                block_ids = set([block.symbol for block in blocks])
                rot_sidechains = len(block_ids.intersection(self.residues)) > 0
                if rot_sidechains:
                    # protein or nucleic acid
                    edges, mask_rotate = get_side_chain_torsion_mask_block(blocks)
                else:
                    edges, mask_rotate = get_segment_torsion_mask(blocks)
                list_of_masks.append({
                    "type": 0 if rot_sidechains else 1,
                    "edges": edges,
                    "mask_rotate": mask_rotate,
                })
            cropped_data["torsion_mask"] = list_of_masks
        if 'modality' in data:
            cropped_data['modality'] = data['modality']
            
        if 'block_embeddings' in data:
            kept_blocks = sorted([0, len(segment0)+1] + kept_old_indices) # [0, len(segment0)+1] for the global blocks
            cropped_data['block_embeddings'] = [data['block_embeddings'][i] for i in kept_blocks]
        elif 'block_embeddings0' in data and 'block_embeddings1' in data:
            kept_blocks0 = [0] + (segment0_keep_indices+1).tolist() # [0] for the global block
            cropped_data['block_embeddings0'] = [data['block_embeddings0'][i] for i in kept_blocks0]
            kept_blocks1 = [0] + (segment1_keep_indices+1).tolist() # [0] for the global block
            cropped_data['block_embeddings1'] = [data['block_embeddings1'][i] for i in kept_blocks1]
        return cropped_data, kept_old_indices

class TorsionNoiseTransform:
    def __init__(self, tor_sigma):
        self.tor_sigma = tor_sigma

    def __call__(self, data, chosen_segment):
        """
        Apply torsion noise to the input data
        Args:
            data: input data
            chosen_segment: segment id
        Returns:
            data with torsion noise
            torsion score [n_tor_edges], None if no torsion edges
            torsion edges [2, n_tor_edges]
        """
        if type(data['X']) == list:
            data['X'] = np.array(data['X'])
        if type(data['block_lengths']) == list:
            data['block_lengths'] = np.array(data['block_lengths']) 
        if type(data['segment_ids']) == list:
            data["segment_ids"] = np.array(data["segment_ids"])

        block_id = np.zeros(len(data['A'])) # [Nu]
        block_id[np.cumsum(data['block_lengths'])[:-1]] = 1
        block_id = np.cumsum(block_id)

        start_block = np.sum(data["segment_ids"] < chosen_segment) + 1 # +1 to skip the global block at the beginning of each segment

        if data['torsion_mask'][chosen_segment]['type'] == 0:
            # sidechain
            torsion_updates = []
            torsion_edges = []

            all_none = all([edges is None for edges in data['torsion_mask'][chosen_segment]['edges']])
            if all_none:
                return data, None, torch.empty((2,0), dtype=torch.long)

            for i, (edges, mask_rotate) in enumerate(zip(data['torsion_mask'][chosen_segment]['edges'], data['torsion_mask'][chosen_segment]['mask_rotate'])):
                if edges is None:
                    continue
                curr_block = start_block + i
                curr_atoms = np.sum(block_id < curr_block)
                num_atoms = data['block_lengths'][curr_block]

                n_rotatable_edges = len(edges)
                torsion_noise = np.random.normal(0, self.tor_sigma, n_rotatable_edges)
                torsion_updates.append(torsion_noise)
                coords = data['X'][curr_atoms:curr_atoms+num_atoms]
                new_coords = modify_conformer_torsion_angles(coords, edges, mask_rotate, torsion_noise)
                data['X'][curr_atoms:curr_atoms+num_atoms] = new_coords
                torsion_edges.append(edges+curr_atoms)
            torsion_updates = np.concatenate(torsion_updates)
            torsion_edges = np.concatenate(torsion_edges, axis=0)
        else:
            # segment
            edges = data['torsion_mask'][chosen_segment]['edges']
            mask_rotate = data['torsion_mask'][chosen_segment]['mask_rotate']

            if edges is None:
                return data, None, torch.empty((2,0), dtype=torch.long)

            start_atoms = np.sum(block_id < start_block)
            
            num_atoms = np.sum(data["block_lengths"][np.logical_and(
                data['segment_ids'] == chosen_segment, 
                np.arange(len(data['segment_ids'])) >= start_block
            )])

            coords = data['X'][start_atoms:start_atoms+num_atoms]
            n_rotatable_edges = len(edges)
            torsion_updates = np.random.normal(0, self.tor_sigma, n_rotatable_edges)

            coords = data['X'][start_atoms:start_atoms+num_atoms]
            new_coords = modify_conformer_torsion_angles(coords, edges, mask_rotate, torsion_updates)
            R, t = rigid_transform_Kabsch_3D(new_coords.T, coords.T)
            new_coords = (R @ new_coords.T + t).T
            data['X'][start_atoms:start_atoms+num_atoms] = new_coords
            torsion_edges = edges+start_atoms

        # update global_block and global_atom
        segment_atoms = np.sum(data['block_lengths'][data['segment_ids'] == chosen_segment])
        global_block = np.sum(data['segment_ids'] < chosen_segment)
        global_atom = np.sum(block_id < global_block)
        data['X'][global_atom] = data['X'][global_atom+1:global_atom+segment_atoms].mean(axis=0)
        # print(f"Torsion angle changes: {torsion_updates}")
        return data, torus_score(torsion_updates, self.tor_sigma), torsion_edges.T
    

class GaussianNoiseTransform:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data, chosen_segment):
        """
        Apply Gaussian noise to the input data
        Args:
            data: input data
            chosen_segment: segment id
        Returns:
            data with Gaussian noise, and the score
        """
        if type(data['X']) == list:
            data['X'] = np.array(data['X'])
        if type(data['block_lengths']) == list:
            data['block_lengths'] = np.array(data['block_lengths']) 
        if type(data['segment_ids']) == list:
            data["segment_ids"] = np.array(data["segment_ids"])

        block_id = np.zeros(len(data['A'])) # [Nu]
        block_id[np.cumsum(data['block_lengths'])[:-1]] = 1
        block_id = np.cumsum(block_id)

        segment_atoms = np.sum(data['block_lengths'][data['segment_ids'] == chosen_segment])
        global_block = np.sum(data['segment_ids'] < chosen_segment)
        global_atom = np.sum(block_id < global_block)

        noise = np.random.normal(0, 1, (segment_atoms-1, 3))
        eps = np.random.uniform(0.1, self.sigma)
        original_coords = copy.deepcopy(data['X'])
        data['X'][global_atom+1:global_atom+segment_atoms] += noise*eps
        data['X'][global_atom] = data['X'][global_atom+1:global_atom+segment_atoms].mean(axis=0)
        atom_score = (original_coords - data['X'])/eps
        return data, atom_score, eps


class GlobalTranslationTransform:
    def __init__(self, tr_sigma):
        self.tr_sigma = tr_sigma
    
    def __call__(self, data, chosen_segment):
        """
        Apply global translation to the input data
        Args:
            data: input data
            chosen_segment: segment id
        Returns:
            data with global translation, and the score
        """
        if type(data['X']) == list:
            data['X'] = np.array(data['X'])
        if type(data['block_lengths']) == list:
            data['block_lengths'] = np.array(data['block_lengths']) 
        if type(data['segment_ids']) == list:
            data["segment_ids"] = np.array(data["segment_ids"])

        block_id = np.zeros(len(data['A'])) # [Nu]
        block_id[np.cumsum(data['block_lengths'])[:-1]] = 1
        block_id = np.cumsum(block_id)

        segment_atoms = np.sum(data['block_lengths'][data['segment_ids'] == chosen_segment])
        global_block = np.sum(data['segment_ids'] < chosen_segment)
        global_atom = np.sum(block_id < global_block)

        eps = np.random.uniform(0.1, self.tr_sigma)
        tr_score = np.random.normal(0, 1, (1, 3))
        data['X'][global_atom:global_atom+segment_atoms] += tr_score * eps
        # print(f"Global translation: {tr_score * eps}")
        return data, np.squeeze(tr_score), eps


def _expansion(theta, sigma, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * np.sin(theta * (l + 1 / 2)) / np.sin(theta / 2)
    return p

def _density(expansion, theta):
    density = expansion * (1 - np.cos(theta)) / np.pi
    density = np.clip(density, 0, 1000)
    return density / density.sum()

def _score(exp, theta, sigma, L=2000):
    dSigma = 0
    for l in range(L):
        hi = np.sin(theta * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(theta * (l + 1 / 2))
        lo = np.sin(theta / 2)
        dlo = 1 / 2 * np.cos(theta / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * (lo * dhi - hi * dlo) / (lo ** 2)
    return dSigma / exp + np.sin(theta) / (1 - np.cos(theta))


class GlobalRotationTransform:
    def __init__(self, rot_sigma, max_theta):
        # Source for SO3 transformation 
        # https://github.com/wengong-jin/DSMBind/blob/master/bindenergy/models/energy.py
        # https://github.com/gcorso/DiffDock/blob/main/utils/so3.py 
        self.theta_range = np.linspace(0.001, max_theta, 100)
        self.sigma = rot_sigma
        self.expansion = _expansion(self.theta_range, self.sigma)
        self.density = _density(self.expansion, self.theta_range)
        self.score = _score(self.expansion, self.theta_range, self.sigma)

    def __call__(self, data, chosen_segment):
        """
        Apply global rotation to the input data
        Args:
            data: input data
            chosen_segment: segment id
        Returns:
            data with global rotation, and the score
        """
    
        tidx = np.random.choice(list(range(100)), p=self.density)
        theta = self.theta_range[tidx]
        w = np.random.normal(0, 1, (1,3))
        hat_w = w / np.linalg.norm(w)

        if type(data['X']) == list:
            data['X'] = np.array(data['X'])
        if type(data['block_lengths']) == list:
            data['block_lengths'] = np.array(data['block_lengths']) 
        if type(data['segment_ids']) == list:
            data["segment_ids"] = np.array(data["segment_ids"])

        block_id = np.zeros(len(data['A'])) # [Nu]
        block_id[np.cumsum(data['block_lengths'])[:-1]] = 1
        block_id = np.cumsum(block_id)

        segment_atoms = np.sum(data['block_lengths'][data['segment_ids'] == chosen_segment])
        global_block = np.sum(data['segment_ids'] < chosen_segment)
        global_atom = np.sum(block_id < global_block)

        coords = data['X'][global_atom+1:global_atom+segment_atoms]
        center = coords.mean(axis=0, keepdims=True)
        R = Rotation.from_rotvec(theta * hat_w)
        new_coords = R.apply(coords - center) + center

        data['X'][global_atom+1:global_atom+segment_atoms] = new_coords
        data['X'][global_atom] = new_coords.mean(axis=0)
        rot_score = hat_w * self.score[tidx]
        # print(f"Global rotation: {theta}")
        return data, np.squeeze(rot_score)