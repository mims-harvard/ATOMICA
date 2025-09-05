import torch
from collections import defaultdict, Counter
from torch_scatter import scatter_mean
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import seaborn as sns
import pickle
import numpy as np
from scipy.spatial import distance_matrix

from atomica.trainers.abs_trainer import Trainer
import atomica.models.masking_model
import atomica.data.dataset_pretrain
from atomica.data.pdb_utils import VOCAB
from atomica.utils.random_seed import setup_seed
from atomica.data.dataset_pretrain import PretrainMaskedDataset, PretrainMaskedTorsionDataset
from atomica.models.masking_model import MaskedNodeModel
import argparse

# TODO: set seed so the same masking is applied
def get_masking_predictions(model, test_dataset, CA_avg_distances, edge_distances, seed):
    setup_seed(seed)
    batch_size = 8
    model = model.to("cuda")
    model.eval()

    preds_list = []
    labels_list = []
    CA_distances_list = []
    intra_distances_list = []
    inter_distances_list = []
    id_list = []

    for i in tqdm(range(0, len(test_dataset), batch_size), total=len(test_dataset)//batch_size):
        with torch.no_grad():
            batch = [test_dataset[j] for j in range(i, min(i + batch_size, len(test_dataset)))]
            batch = PretrainMaskedDataset.collate_fn(batch)
            masked_blocks = batch['masked_blocks'].detach().cpu().nonzero().squeeze()
            batch_ids = sum([[x]*item_len for x, item_len in enumerate(batch['lengths'])], [])
            for masked_block in masked_blocks:
                batch_id = batch_ids[masked_block]
                block_id = masked_block - batch['lengths'][:batch_id].sum()
                item_id = i + batch_id
                CA_distances_list.append(CA_avg_distances[item_id].get(int(block_id), None))
                intra_distances, inter_distances = edge_distances[item_id]
                if len(intra_distances) != batch['lengths'][batch_id]:
                    print(block_id, item_id)
                intra_distances_list.append(intra_distances[block_id].item())
                inter_distances_list.append(inter_distances[block_id].item())
                id_list.append(test_dataset.indexes[i+batch_id]['id'])
            batch = Trainer.to_device(batch, "cuda")
            preds = model.infer(batch).detach().cpu()
            labels = batch['masked_labels'].detach().cpu()
            preds_list.append(preds)
            labels_list.append(labels)

    preds_list = torch.cat(preds_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    return id_list, preds_list, labels_list, CA_distances_list, intra_distances_list, inter_distances_list

def main(modality, data_file, model_ckpt, model1_ckpt, model_is_mask_only, model1_is_mask_only, output_path):
    VOCAB.load_tokenizer('PS_300')
    modality = modality
    test_dataset = PretrainMaskedDataset(
        data_file=data_file,
        mask_proportion= 0.1,
        mask_token= VOCAB.symbol_to_idx(VOCAB.MASK),
        vocab_to_mask= [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
        atom_mask_token= VOCAB.get_atom_mask_idx(),
    )

    if model_is_mask_only:
        model: MaskedNodeModel = torch.load(model_ckpt)
    else:
        model = MaskedNodeModel.load_from_pretrained(model_ckpt, 
                                                num_masked_block_classes=len(test_dataset.vocab_to_mask))
    if model1_is_mask_only:
        model1: MaskedNodeModel = torch.load(model1_ckpt)
    else:
        model1 = MaskedNodeModel.load_from_pretrained(model1_ckpt, 
                                                    num_masked_block_classes=len(test_dataset.vocab_to_mask))

    num_params = sum(p.numel() for p in model.parameters())
    num_params1 = sum(p.numel() for p in model1.parameters())
    print("Num params", num_params, num_params1)

    CA_avg_distances = {}
    k = 8
    for idx, item in enumerate(test_dataset.data):
        CA_indexes = [i for i, x in enumerate(item['data']['atom_positions']) if VOCAB.idx_to_atom_pos(x) == 'A']
        block_id = sum([[i]*block_len for i, block_len in enumerate(item['data']['block_lengths'])], [])
        CA_block_id = [block_id[i] for i in CA_indexes]
        CA_x = [item['data']['X'][i] for i in CA_indexes]
        if len(CA_x) == 0:
            CA_avg_distances[idx] = {}
            continue
        CA_dist = distance_matrix(CA_x, CA_x)
        sorted_CA_dist = np.sort(CA_dist, axis=1)
        avg_distances = np.mean(sorted_CA_dist[:, 1:k+1], axis=1)
        CA_avg_distances[idx] = dict(zip(CA_block_id, avg_distances))
    
    edge_distances = {}
    for idx, item in enumerate(test_dataset.data):
        B = torch.tensor(item['data']['B'], dtype=torch.long)
        batch_id = torch.zeros_like(B)
        segment_ids = torch.tensor(item['data']['segment_ids'], dtype=torch.long)
        block_id = torch.tensor(sum([[i]*block_len for i, block_len in enumerate(item['data']['block_lengths'])], []), dtype=torch.long)
        X_atom = torch.tensor(item['data']['X'], dtype=torch.float)
        X = scatter_mean(X_atom, block_id, dim=0)
        intra_edges, inter_edges, global_normal_edges, global_global_edges, _ = model.edge_constructor(B, batch_id, segment_ids, X=X, block_id=torch.arange(len(B)))

        diff = X.unsqueeze(1) - X.unsqueeze(0)
        dist_squared = torch.sum(diff ** 2, dim=-1)
        dist_matrix = torch.sqrt(dist_squared)

        mean_intra_dist = scatter_mean(dist_matrix[intra_edges[0], intra_edges[1]], intra_edges[0], dim=0, out=torch.zeros_like(B, dtype=torch.float))
        mean_inter_dist = scatter_mean(dist_matrix[inter_edges[0], inter_edges[1]], inter_edges[0], dim=0, out=torch.zeros_like(B, dtype=torch.float))
        edge_distances[idx] = mean_intra_dist, mean_inter_dist
    
    results = []
    for seed in [0, 1, 2, 3, 4]:
        id_list, preds_list, labels_list, CA_distances_list, intra_distances_list, inter_distances_list = get_masking_predictions(
            model, test_dataset, CA_avg_distances, edge_distances, seed)
        results.append((id_list, preds_list, labels_list, CA_distances_list, intra_distances_list, inter_distances_list))
    
    results1 = []
    for seed in [0, 1, 2, 3, 4]:
        id_list1, preds_list1, labels_list1, CA_distances_list1, intra_distances_list1, inter_distances_list1 = get_masking_predictions(
            model1, test_dataset, CA_avg_distances, edge_distances, seed)
        results1.append((id_list1, preds_list1, labels_list1, CA_distances_list1, intra_distances_list1, inter_distances_list1))
    

    with open(output_path, "wb") as f:
        pickle.dump((model_ckpt, results, model1_ckpt, results1), f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="PP", help="Modality")
    parser.add_argument("--data_file", type=str, help="Data file")
    parser.add_argument("--model_ckpt", type=str, help="Model (single modality) checkpoint")
    parser.add_argument("--model1_ckpt", type=str, help="Model1 (InteractNN) checkpoint")
    parser.add_argument("--model_is_mask_only", action="store_true", help="Model is mask only", default=False)
    parser.add_argument("--model1_is_mask_only", action="store_true", help="Model is mask only", default=False)
    parser.add_argument("--output_path", type=str, help="Path to save results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.modality, args.data_file, args.model_ckpt, args.model1_ckpt, args.model_is_mask_only, args.model1_is_mask_only, args.output_path)