import os
import sys

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
# print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.sm_pdb_to_blocks import sm_pdb_to_blocks
from data.dataset import VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.dataset import BlockGeoAffDataset, PDBBindBenchmark, DynamicBatchWrapper, PDBDataset
from data.atom3d_dataset import LBADataset
import models
import torch
from trainers.abs_trainer import Trainer
import importlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models import DenoisePretrainModel, AffinityPredictor
from torch_scatter import scatter_sum


def get_embeddings(dataset, model, out_dir):
    train_loader = DataLoader(dataset, batch_size=1,
                                num_workers=4,
                                shuffle=False, # for same ordering between pretraining and finetuning runs
                                collate_fn = dataset.collate_fn)
    model = model.to("cuda")
    block_embeddings = []
    block_id = []
    atom_embeddings = []
    atom_id = []
    graph_embeddings = []
    graph_unit_embeddings = []

    for idx, batch in tqdm(enumerate(train_loader)):
        try:
            model.eval()
            with torch.no_grad():
                batch = Trainer.to_device(batch, "cuda")
                pred_binding_affinity, output = model.infer(batch, extra_info=True)
                block_embedding = output.block_repr.detach().clone().cpu()
                atom_embedding = output.unit_repr.detach().clone().cpu()
                graph_embedding = output.graph_repr.detach().clone().cpu()
                graph_unit_embedding = output.graph_unit_repr.detach().clone().cpu()
                del output
                # embedding = embedding[batch['B'].cpu() > 3] # remove special tokens
                block_embeddings.append(block_embedding.numpy())
                block_id.append(batch['B'].cpu().numpy()) 
                atom_embeddings.append(atom_embedding.numpy())
                atom_id.append(batch['A'].cpu().numpy())
                graph_embeddings.append(graph_embedding.numpy())
                graph_unit_embeddings.append(graph_unit_embedding.numpy())
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"Out of memory at {idx}, num_blocks: {batch['B'].shape}, num_atoms: {batch['A'].shape}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if len(block_embeddings) == 0:
        print("No embeddings, OOM")
        return

    block_embeddings = np.concatenate(block_embeddings, axis=0)
    block_id = np.concatenate(block_id, axis=0)
    atom_embeddings = np.concatenate(atom_embeddings, axis=0)
    atom_id = np.concatenate(atom_id, axis=0)
    graph_embeddings = np.concatenate(graph_embeddings, axis=0)
    graph_unit_embeddings = np.concatenate(graph_unit_embeddings, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/block_embeddings.npy", "wb") as f:
        np.save(f, block_embeddings)
    with open(f"{out_dir}/block_id.npy", "wb") as f:
        np.save(f, block_id)
    with open(f"{out_dir}/atom_embeddings.npy", "wb") as f:
        np.save(f, atom_embeddings)
    with open(f"{out_dir}/atom_id.npy", "wb") as f:
        np.save(f, atom_id)
    with open(f"{out_dir}/graph_embeddings.npy", "wb") as f:
        np.save(f, graph_embeddings)
    with open(f"{out_dir}/graph_unit_embeddings.npy", "wb") as f:
        np.save(f, graph_unit_embeddings)


if __name__ == "__main__":
    pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-torsion/version_1/checkpoint/epoch11_step188074.ckpt"
    # pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-gaussian/version_0/checkpoint/epoch1_step26644.ckpt"
    model = AffinityPredictor.load_from_pretrained(pretrain_ckpt)
    out_type = "pretrained"

    # finetune_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_112/checkpoint/epoch144_step61190.ckpt"
    # model = torch.load(finetune_ckpt, map_location="cpu")
    # out_type = "finetuned"
    
    # out_dir = f"case_studies/visualise_nodes/{out_type}_embeddings1/PLA30"
    # dataset = PDBBindBenchmark("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/PDBBind/processed_PS_300/identity30/valid.pkl")
    # dataset = DynamicBatchWrapper(dataset, 500)
    # get_embeddings(dataset, model, out_dir)
    # print("Finished PLA30")

    out_dir = f"case_studies/visualise_nodes/{out_type}_embeddings_all/CSD_valid_PS_300"
    dataset = PDBBindBenchmark("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/CSD/valid_PS_300.pkl")
    dataset = DynamicBatchWrapper(dataset, 500)
    get_embeddings(dataset, model, out_dir)
    print("Finished CSD")

    # out_dir = f"case_studies/visualise_nodes/{out_type}_embeddings1/BioLiP"
    # dataset = PDBBindBenchmark("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/BioLiP/processed-QBioLiP/train_subsampled_2000.pkl")
    # dataset = DynamicBatchWrapper(dataset, 500)
    # get_embeddings(dataset, model, out_dir)

    datasets = os.listdir("./datasets/BioLiP/processed-QBioLiP/")
    for dataset_name in datasets:
        if not dataset_name.startswith("QBioLiP") or not dataset_name.endswith(".pkl") or "train" in dataset_name:
            continue
        datatype = dataset_name.split("_")[1]
        dataset = PDBBindBenchmark(f"datasets/BioLiP/processed-QBioLiP/{dataset_name}")
        out_dir = f"case_studies/visualise_nodes/{out_type}_embeddings_all/{dataset_name[:-len('.pkl')]}"
        os.makedirs(out_dir, exist_ok=True)
        dataset = DynamicBatchWrapper(dataset, 500)
        get_embeddings(dataset, model, out_dir)
        print(f"Finished {datatype}. Saved to {out_dir}")

    # out_dir = "/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/wdr_proteins/embeddings"
    # dataset = PDBDataset("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/wdr_proteins/wdr_interfaces.pkl")
    # dataset = DynamicBatchWrapper(dataset, 500)
    # get_embeddings(dataset, model, out_dir)
    