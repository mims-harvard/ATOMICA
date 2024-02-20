from tqdm import tqdm
import os
import pickle
import random
import argparse

def main(processed_csd_dir):
    train_dataset = []
    valid_dataset = []
    for fname in tqdm(os.listdir(processed_csd_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(f"{processed_csd_dir}/{fname}", "rb") as f:
            data = pickle.load(f)
            if not fname.startswith("QBioLiP"):
                continue
            if "train" in fname:
                train_dataset.extend(data)
            else:
                valid_dataset.extend(data)
    
    print(f"Train dataset size: {len(train_dataset)}")
    with open(f"{processed_csd_dir}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"Valid dataset size: {len(valid_dataset)}")
    with open(f"{processed_csd_dir}/valid.pkl", "wb") as f:
        pickle.dump(valid_dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir', type=str, default='./datasets/CSD')
    args = parser.parse_args()
    main(args.processed_data_dir)