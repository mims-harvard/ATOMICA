import pandas as pd
import torch
from tqdm import tqdm
import fm
import os
import numpy as np

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"
os.environ["TORCH_HOME"] = "/n/netscratch/mzitnik_lab/Lab/afang/torch_cache"

def get_embeddings(task_name):
    df = pd.read_parquet(f'{DATA_DIR}/{task_name}/{task_name}_sequences.parquet')
    df['id'] = df['pdb_id'] + '_' + df['chain_id']
    assert df['id'].nunique() == len(df), "id is not unique"

    data = list(zip(df['id'].tolist(), df['seq'].tolist()))

    print("Loading model...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to("cuda")

    print("Model loaded")

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    all_embeddings = []

    # there will be a <bos>=0, <eos>=2 and padding=1 tokens

    batch_size = 4
    for i in tqdm(range(0, len(data), batch_size), desc="Embedding data", total=len(data)//batch_size+1):
        batch = batch_tokens[i:i+batch_size]
        all_token_embeddings = []
        for seq_start in range(0, batch.shape[1], 512):
            with torch.no_grad():
                results = model(batch[:, seq_start:seq_start+512].to("cuda"), repr_layers=[12])
            token_embeddings = results["representations"][12].cpu() # [batch_size, max_len, 640]
            all_token_embeddings.append(token_embeddings)
        token_embeddings = torch.cat(all_token_embeddings, dim=1)
        all_embeddings.append(token_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    with open(f"{DATA_DIR}/{task_name}/{task_name}_embeddings_rnafm.npy", "wb") as f:
        np.save(f, all_embeddings.numpy())

    with open(f"{DATA_DIR}/{task_name}/{task_name}_tokens_rnafm.npy", "wb") as f:
        np.save(f, batch_tokens.numpy())
    
    print(f"Embeddings saved to {DATA_DIR}/{task_name}/{task_name}_embeddings_rnafm.npy")

if __name__ == "__main__":
    task_names = [
        "RNA_Protein",
        "RNA_Site",
        "RNA_Ligand",
        "RNAGo",
    ]
    for task_name in task_names:
        get_embeddings(task_name)