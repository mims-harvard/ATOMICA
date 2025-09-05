import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA

from atomica.data.pdb_utils import VOCAB
VOCAB.load_tokenizer('PS_300')

embedding_dir = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/embeddings/version54_epoch73/"

atom_embeddings = defaultdict(list)
block_embeddings = defaultdict(list)

for embedding_file in os.listdir(embedding_dir):
    if not embedding_file.endswith(".pkl"):
        continue
    with open(os.path.join(embedding_dir, embedding_file), "rb") as f:
        embeddings = pickle.load(f)
    for item in tqdm(embeddings, total=len(embeddings), desc=embedding_file):
        for atom_id, atom_embedding in zip(item['atom_id'], item['atom_embedding']):
            atom_embeddings[atom_id].append(atom_embedding)
        for block_id, block_embedding in zip(item['block_id'], item['block_embedding']):
            block_embeddings[block_id].append(block_embedding)

mean_atom_embeddings_dict = {k: np.array(v).mean(axis=0) for k, v in atom_embeddings.items()}
mean_block_embeddings_dict = {k: np.array(v).mean(axis=0) for k, v in block_embeddings.items()}

mean_atom_embeddings = np.array(list(mean_atom_embeddings_dict.values()))
mean_block_embeddings = np.array(list(mean_block_embeddings_dict.values()))
atom_ids = np.array(list(mean_atom_embeddings_dict.keys()))
block_ids = np.array(list(mean_block_embeddings_dict.keys()))

pca = PCA(n_components=2)
atom_pca = pca.fit_transform(mean_atom_embeddings)
block_pca = pca.fit_transform(mean_block_embeddings)

atom_output = {
    "x": atom_pca[:, 0].tolist(),
    "y": atom_pca[:, 1].tolist(),
    "atom_id": atom_ids.tolist(),
    "atom_type": [VOCAB.idx_to_atom(i) for i in atom_ids]
}

block_output = {
    "x": block_pca[:, 0].tolist(),
    "y": block_pca[:, 1].tolist(),
    "block_id": block_ids.tolist(),
    "block_type": [VOCAB.idx_to_abrv(i) for i in block_ids]
}

with open("atom_pca.pkl", "wb") as f:
    pickle.dump(atom_output, f)
with open("block_pca.pkl", "wb") as f:
    pickle.dump(block_output, f)