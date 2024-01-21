from tqdm import tqdm
import os
import pickle
import random
import argparse

def main(processed_csd_dir):
    processed_data = []
    for f in tqdm(os.listdir(processed_csd_dir)):
        with open(f"{processed_csd_dir}/{f}", "rb") as f:
            data = pickle.load(f)
            processed_data.extend(data)
    
    random.shuffle(processed_data)
    train_dataset = processed_data[:-10000]
    valid_dataset = processed_data[-10000:]
    with open(f"{processed_csd_dir}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(f"{processed_csd_dir}/valid.pkl", "wb") as f:
        pickle.dump(valid_dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir', type=str, default='./datasets/CSD')
    args = parser.parse_args()
    main(args.processed_data_dir)