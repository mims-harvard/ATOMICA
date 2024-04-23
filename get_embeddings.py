from tqdm import tqdm
import pickle
from data.dataset import PDBDataset
from models.prediction_model import PredictionModel
from trainers.abs_trainer import Trainer

def main(args):
    dataset = PDBDataset(args.data_path)
    model = PredictionModel.load_from_pretrained(args.model_ckpt)
    model = model.to("cuda")
    batch_size = args.batch_size

    embeddings = []
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Embedding data", total=len(dataset)//batch_size+1):
        items = dataset.data[idx:min(idx+batch_size, len(dataset))]

        outputs = []
        for item in items:
            outputs.append({"id": item["id"]})
        batch = dataset.collate_fn([item["data"] for item in items])
        batch = Trainer.to_device(batch, "cuda")
        return_obj = model.infer(batch)
        
        curr_block = 0
        curr_atom = 0
        for i, item in enumerate(items):
            num_blocks = len(item["data"]["B"])
            num_atoms = len(item["data"]["A"])

            outputs[i]["graph_embedding"] = return_obj.graph_repr[i].detach().cpu().numpy()
            outputs[i]["block_embedding"] = return_obj.block_repr[curr_block: curr_block + num_blocks].detach().cpu().numpy()
            outputs[i]["atom_embedding"] = return_obj.unit_repr[curr_atom: curr_atom + num_atoms].detach().cpu().numpy()
            outputs[i]["block_id"] = item["data"]["B"]
            outputs[i]["atom_id"] = item["data"]["A"]

            curr_block += num_blocks
            curr_atom += num_atoms

        embeddings.extend(outputs)
    
    with open(args.output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saving processed data to {args.output_path}. Total of {len(embeddings)} items.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)