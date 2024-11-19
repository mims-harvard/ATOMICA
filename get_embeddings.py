from tqdm import tqdm
import pickle
from data.dataset import PDBDataset, ProtInterfaceDataset
from models.prediction_model import PredictionModel
from models.prot_interface_model import ProteinInterfaceModel
from trainers.abs_trainer import Trainer
import torch

def main(args):

    # # Filter out large items
    # too_large = []
    # new_data, new_indexes = [], []
    # for item in dataset.data:
    #     if len(item["data"]['B']) <= 500:
    #         new_data.append(item)
    #         new_indexes.append(item["id"])
    #     else:
    #         too_large.append(item["id"])
    # dataset.data = new_data
    # dataset.indexes = new_indexes
    # print(f"Removed {len(too_large)} items that are too large. Remaining {len(dataset)} items.")

    model = torch.load(args.model_ckpt)
    if isinstance(model, ProteinInterfaceModel):
        print("Model is ProteinInterfaceModel, extracting prot_model.")
        model = model.prot_model
        dataset = ProtInterfaceDataset(args.data_path)
    else:
        dataset = PDBDataset(args.data_path)

    # model = PredictionModel.load_from_pretrained(args.model_ckpt)
    model = model.to("cuda")
    batch_size = args.batch_size

    embeddings = []
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Embedding data", total=len(dataset)//batch_size+1):
        items = dataset.data[idx:min(idx+batch_size, len(dataset))]

        outputs = []
        try:
            for item in items:
                outputs.append({"id": item["id"]})
            if isinstance(dataset, ProtInterfaceDataset):
                batch_items = [item["prot_data"] for item in items]
            else:
                batch_items = [item["data"] for item in items]
            batch = PDBDataset.collate_fn(batch_items)
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
        except Exception as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print("CUDA out of memory, reducing batch size to 1 for this batch.")
                outputs = []
                # go through the batch one by one
                for item in items:
                    try:
                        output = {"id": item["id"]}
                        batch = PDBDataset.collate_fn([item["data"] if not isinstance(dataset, ProtInterfaceDataset) else item["prot_data"]])
                        batch = Trainer.to_device(batch, "cuda")
                        return_obj = model.infer(batch)
                        output["graph_embedding"] = return_obj.graph_repr[0].detach().cpu().numpy()
                        output["block_embedding"] = return_obj.block_repr.detach().cpu().numpy()
                        output["atom_embedding"] = return_obj.unit_repr.detach().cpu().numpy()
                        output["block_id"] = item["data"]["B"]
                        output["atom_id"] = item["data"]["A"]
                        outputs.append(output)
                    except Exception as e:
                        print(f"Error processing item {item['id']}: {e}")
                        torch.cuda.empty_cache()
                        continue
            else:
                import pdb; pdb.set_trace()
                raise e
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