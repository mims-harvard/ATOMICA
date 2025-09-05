
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, auc

# Imports for interface evaluation

from atomica.data.dataset import PDBDataset
from atomica.trainers.abs_trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ZN')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def inference(model_ckpt, dataset_path, outpath, batch_size = 16):
    model = torch.load(model_ckpt, map_location='cuda')
    with open(dataset_path, "rb") as f:
        test_dataset = pickle.load(f)
    predicted = []
    for i in tqdm(range(0, len(test_dataset), batch_size), total=len(test_dataset)//batch_size):
        end = min(i+batch_size, len(test_dataset))
        batch = PDBDataset.collate_fn([test_dataset[j]['data'] for j in range(i, end)])
        batch = Trainer.to_device(batch, 'cuda')
        output = model.infer(batch).detach().cpu()
        predicted.append(output)
    predicted = torch.cat(predicted, dim=0).squeeze()
    if 'label' in test_dataset[0]:
        labels = np.array([int(test_dataset[i]['label']) for i in range(len(test_dataset))])
        precision, recall, _ = precision_recall_curve(labels, predicted)
        auprc = auc(recall, precision)
        print(f"AUPRC: {auprc} baseline: {np.mean(labels)}")
    else:
        labels = [pd.NA]*len(test_dataset)

    output = {
        "predicted": predicted,
        "label": labels,
        "id": [test_dataset[i]['id'] for i in range(len(test_dataset))],
    }
    output = pd.DataFrame(output)
    output.to_csv(outpath, index=False)
    print(f"Saved to {outpath}")

if __name__ == "__main__":
    args = parse_args()
    topk_file = os.path.join(args.model_dir, "checkpoint/topk_map.txt")
    with open(topk_file, 'r') as f:
        topk_map = f.readlines()
    best_ckpt = topk_map[0].split()[1]
    inference(best_ckpt, args.data_path, args.output_path)