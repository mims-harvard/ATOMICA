import os
import numpy as np
import pickle
from scipy.stats import spearmanr
from atomica.data.pdb_utils import VOCAB

def get_frequency_vector(sample, fragmentation_type='PS_300'):
    """
    Given a sample, return a frequency vector of blocks in the sample
    """
    VOCAB.load_tokenizer(fragmentation_type)
    num_block_types = len(VOCAB.abrv2idx)
    blocks = np.array(sample["data"]["B"])
    unique_values, counts = np.unique(blocks, return_counts=True)
    frequency_vector = np.zeros(num_block_types)
    frequency_vector[unique_values] = counts
    return frequency_vector

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    return parser.parse_args()


def main(args):
    freq_vectors = []
    indexes = []
    ids = []
    for dataset_path in args.dataset:
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        for idx, sample in enumerate(dataset):
            freq_vector = get_frequency_vector(sample)
            freq_vectors.append(freq_vector)
            indexes.append(f'{dataset_name}_{idx}')
            ids.append(sample["id"])
    
    freq_vectors = np.array(freq_vectors)

    with open(args.output_path, "wb") as f:
        pickle.dump({
            "freq_vectors": freq_vectors,
            "indexes": indexes,
            "ids": ids,
            # "spearmanr": spearmanr(freq_vectors, axis=1),
        }, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
        
        
