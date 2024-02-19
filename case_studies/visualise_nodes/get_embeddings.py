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
from data.dataset import BlockGeoAffDataset, PDBBindBenchmark, DynamicBatchWrapper
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


def get_embeddings(dataset, model, out_dir):
    dataset = DynamicBatchWrapper(dataset, 400)
    train_loader = DataLoader(dataset, batch_size=1,
                                num_workers=1,
                                shuffle=False, # for same ordering between pretraining and finetuning runs
                                collate_fn = dataset.collate_fn)
    model = model.to("cuda")
    block_embeddings = []
    block_id = []
    atom_embeddings = []
    atom_id = []
    graph_embeddings = []
    for idx, batch in tqdm(enumerate(train_loader)):
        model.eval()
        with torch.no_grad():
            batch = Trainer.to_device(batch, "cuda")
            if isinstance(model, AffinityPredictor):
                pred_binding_affinity, output = model.infer(batch, extra_info=True)
            else:
                output = model(Z=batch['X'], B=batch['B'], A=batch['A'],
                            atom_positions=batch['atom_positions'],
                            block_lengths=batch['block_lengths'],
                            lengths=batch['lengths'],
                            segment_ids=batch['segment_ids'],
                            label=None,
                            return_loss=True)
            block_embedding = output.block_repr.detach().clone().cpu()
            atom_embedding = output.unit_repr.detach().clone().cpu()
            graph_embedding = output.graph_repr.detach().clone().cpu()
            del output
            # embedding = embedding[batch['B'].cpu() > 3] # remove special tokens
            block_embeddings.append(block_embedding.numpy())
            block_id.append(batch['B'].cpu().numpy()) 
            atom_embeddings.append(atom_embedding.numpy())
            atom_id.append(batch['A'].cpu().numpy())
            graph_embeddings.append(graph_embedding.numpy())

    block_embeddings = np.concatenate(block_embeddings, axis=0)
    block_id = np.concatenate(block_id, axis=0)
    atom_embeddings = np.concatenate(atom_embeddings, axis=0)
    atom_id = np.concatenate(atom_id, axis=0)
    graph_embeddings = np.concatenate(graph_embeddings, axis=0)

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


if __name__ == "__main__":
    # pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-global/version_72/checkpoint/epoch4_step397797.ckpt"
    # pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-global/version_76/checkpoint/epoch4_step397797.ckpt"
    pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_112/checkpoint/epoch144_step61190.ckpt"
    # pretrain_ckpt = '/n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_104/checkpoint/epoch129_step54860.ckpt'
    model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')

    # pretrain_ckpt = "/n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_98/checkpoint/epoch109_step46420.ckpt" # finetuned from the 4 day long pretraining run
    # # "/n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_89/checkpoint/epoch69_step29470.ckpt" # finetuned from QBioLIP
    # model: AffinityPredictor = torch.load(pretrain_ckpt, map_location='cpu')

    # dataset = LBADataset("/n/holylabs/LABS/mzitxnik_lab/Users/afang/GET/datasets/LBA/split-by-sequence-identity-30/train")
    # dataset = PDBBindBenchmark("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/BioLiP/processed-QBioLiP/train_subsampled_2000.pkl")
    # dataset = PDBBindBenchmark("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/CSD/train_subsampled_2000.pkl")

    datasets = os.listdir("datasets/BioLiP/processed-QBioLiP/subsampled")
    for dataset in datasets:
        datatype = dataset.split("_")[1]
        dataset = PDBBindBenchmark(f"datasets/BioLiP/processed-QBioLiP/subsampled/{dataset}")
        out_dir = f"case_studies/visualise_nodes/BioLiP/data_pretrained_{datatype}"
        os.makedirs(out_dir, exist_ok=True)
        get_embeddings(dataset, model, out_dir)
    
    out_dir = "case_studies/visualise_nodes/PLA30/data_finetuned"

