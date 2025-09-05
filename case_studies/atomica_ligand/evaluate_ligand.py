import numpy as np
import json
import os
import pandas as pd
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc

# Get the absolute path to the ATOMICA root directory
current_file = os.path.abspath(__file__)
atomica_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, atomica_root)

from atomica.data.dataset import LabelledPDBDataset
from atomica.models.classifier_model import ClassifierModel
from atomica.trainers.abs_trainer import Trainer

batch_size = 16

model = ClassifierModel.load_from_config_and_weights(
    "/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/checkpoints/ligand/HEM/HEM_v1_config.json", 
    "/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/checkpoints/ligand/HEM/HEM_v1.pt",
)
model.to('cuda')

dataset = LabelledPDBDataset("/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/atomica_ligand/data/PL/HEM/HEM_sequence_30_test.pkl")

ground_truth = [x['label'] for x in dataset.data]
ground_truth = np.array(ground_truth)

predictions = []
for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size):
    batch = LabelledPDBDataset.collate_fn([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
    batch = Trainer.to_device(batch, 'cuda')

    batch_predictions = []
    prediction = model.infer(batch).detach().cpu()
    batch_predictions.append(prediction)
    batch_predictions = torch.mean(torch.stack(batch_predictions), dim=0).detach().cpu().numpy()
    predictions.append(batch_predictions)
predictions = np.concatenate(predictions, axis=0)

recall, precision, _ = precision_recall_curve(ground_truth, predictions)
auprc = auc(precision, recall)
print("Num samples: ", len(predictions))
print("Baseline AUPRC: ", np.mean(ground_truth))
print(f"AUPRC: {auprc:.4f}") # ATOMICA AUPRC: 0.6596
