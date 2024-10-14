import os
import sys
import json
from tqdm import tqdm
import numpy as np
import torch

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
# print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.dataset import DynamicBatchWrapper, PDBDataset
from trainers.abs_trainer import Trainer
from torch.utils.data import DataLoader
from models import PredictionModel


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

    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Get embeddings"):
        try:
            model.eval()
            with torch.no_grad():
                batch = Trainer.to_device(batch, "cuda")
                output = model.infer(batch)
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


def get_embeddings_with_graph_id(dataset, model, out_dir, batch_size=8):
    model = model.to("cuda")
    block_embeddings = []
    block_id = []
    atom_embeddings = []
    atom_id = []
    graph_embeddings = []
    graph_id = []

    for idx in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size, desc="Get embeddings"):
        batch = dataset.collate_fn([dataset[i] for i in range(idx, min(idx+batch_size, len(dataset)))])
        graph_id.extend(dataset.indexes[idx:min(idx+batch_size, len(dataset))])
        try:
            model.eval()
            with torch.no_grad():
                batch = Trainer.to_device(batch, "cuda")
                output = model.infer(batch)
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
    with open(f"{out_dir}/graph_id.txt", "w") as f:
        for gid in graph_id:
            f.write(f"{gid}\n")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    parser.add_argument("--with_graph_id", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = PredictionModel.load_from_pretrained(args.pretrain_ckpt)

    for datasetf in args.dataset:
        dataset = PDBDataset(datasetf)
        data_name = os.path.basename(datasetf)[:-len(".pkl")]
        out_path = f"{args.out_dir}/{data_name}"
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/config.json", "w") as f:
            json.dump(vars(args), f)
        if args.with_graph_id:
            get_embeddings_with_graph_id(dataset, model, out_path)
        else:
            dataset = DynamicBatchWrapper(dataset, 500)
            get_embeddings(dataset, model, out_path)
        print(f"Finished {datasetf}. Saved to {out_path}")