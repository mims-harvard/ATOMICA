from tqdm import tqdm
import pickle
from .data.dataset import PDBDataset, ProtInterfaceDataset
from .models.prediction_model import PredictionModel
from .models.prot_interface_model import ProteinInterfaceModel
from .trainers.abs_trainer import Trainer
from .utils import pickled_checkpoint_error
import torch
import json
import pandas as pd

def main(args):
    """Generate embeddings for molecular structures using ATOMICA models.

    This function loads a pre-trained ATOMICA model and generates embeddings
    for molecular structures from a dataset. The embeddings include graph-level,
    block-level (residue/fragment), and atom-level representations.

    Args:
        args: Namespace or object with the following attributes:
            - model_config (str): Path to model config JSON file
            - model_weights (str): Path to model weights file (a state dict)
            - data_path (str): Path to input data file (.pkl, .parquet, or .json)
            - output_path (str): Path to save output embeddings (.pkl or .parquet)
            - batch_size (int): Batch size for processing (default: 4)
            - device (str, optional): 'cuda', 'cpu', or 'auto' (default: 'auto')

    Returns:
        None. Saves embeddings to the file specified in args.output_path.

        For pickle output: Saves list of dicts with keys:
            - 'id': Structure identifier
            - 'graph_embedding': Whole graph/complex embedding
            - 'block_embedding': Per-residue/fragment embeddings
            - 'atom_embedding': Per-atom embeddings
            - 'block_id': Block identifiers
            - 'atom_id': Atom identifiers

        For parquet output: Same structure saved as a pandas DataFrame.

    Example:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     model_config='pretrain_model_config.json',
        ...     model_weights='pretrain_model_weights.pt',
        ...     data_path='structures.pkl',
        ...     output_path='embeddings.pkl',
        ...     batch_size=4
        ... )
        >>> main(args)
    """
    if args.model_ckpt:
        raise pickled_checkpoint_error(args.model_ckpt, "--model_config", "--model_weights")
    if args.model_config and args.model_weights:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        if model_config['model_type'] == 'PredictionModel' or model_config['model_type'] == 'DenoisePretrainModel':
            model = PredictionModel.load_from_config_and_weights(args.model_config, args.model_weights)
        elif model_config['model_type'] == 'ProteinInterfaceModel':
            model = ProteinInterfaceModel.load_from_config_and_weights(args.model_config, args.model_weights)
        else:
            raise NotImplementedError(f"Model type {model_config['model_type']} not implemented")
    else:
        raise ValueError("Both --model_config and --model_weights are required.")

    if isinstance(model, ProteinInterfaceModel):
        print("Model is ProteinInterfaceModel, extracting prot_model.")
        model = model.prot_model
        dataset = ProtInterfaceDataset(args.data_path)
    else:
        dataset = PDBDataset(args.data_path)
    
    device = getattr(args, "device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Check that the installed torch build matches your CUDA driver "
            "(see setup/README.md), or pass --device cpu."
        )
    print(f"Running on device: {device}")
    model = model.to(device)
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
            batch = Trainer.to_device(batch, device)
            return_obj = model.infer(batch)
            
            curr_block = 0
            curr_atom = 0
            for i, item in enumerate(items):
                # Use the actual data that was passed to the model for correct dimensions
                if isinstance(dataset, ProtInterfaceDataset):
                    actual_data = item["prot_data"]
                else:
                    actual_data = item["data"]

                num_blocks = len(actual_data["B"])
                num_atoms = len(actual_data["A"])

                outputs[i]["graph_embedding"] = return_obj.graph_repr[i].detach().cpu().numpy()
                outputs[i]["block_embedding"] = return_obj.block_repr[curr_block: curr_block + num_blocks].detach().cpu().numpy()
                outputs[i]["atom_embedding"] = return_obj.unit_repr[curr_atom: curr_atom + num_atoms].detach().cpu().numpy()
                outputs[i]["block_id"] = actual_data["B"]
                outputs[i]["atom_id"] = actual_data["A"]

                curr_block += num_blocks
                curr_atom += num_atoms
        except Exception as e:
            if "CUDA out of memory" in str(e):
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print("CUDA out of memory, reducing batch size to 1 for this batch.")
                outputs = []
                # go through the batch one by one
                for item in items:
                    try:
                        output = {"id": item["id"]}
                        # Use the actual data that was passed to the model
                        if isinstance(dataset, ProtInterfaceDataset):
                            actual_data = item["prot_data"]
                        else:
                            actual_data = item["data"]

                        batch = PDBDataset.collate_fn([actual_data])
                        batch = Trainer.to_device(batch, device)
                        return_obj = model.infer(batch)
                        output["graph_embedding"] = return_obj.graph_repr[0].detach().cpu().numpy()
                        output["block_embedding"] = return_obj.block_repr.detach().cpu().numpy()
                        output["atom_embedding"] = return_obj.unit_repr.detach().cpu().numpy()
                        output["block_id"] = actual_data["B"]
                        output["atom_id"] = actual_data["A"]
                        outputs.append(output)
                    except Exception as e:
                        print(f"Error processing item {item['id']}: {e}")
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        continue
            else:
                raise e
        embeddings.extend(outputs)
    
    if args.output_path.endswith('.parquet'):
        # Save as parquet file
        # Convert numpy arrays to lists for parquet compatibility
        import numpy as np
        embeddings_parquet = []
        for emb in embeddings:
            emb_copy = {
                "id": emb["id"],
                "graph_embedding": emb["graph_embedding"].tolist() if isinstance(emb["graph_embedding"], np.ndarray) else emb["graph_embedding"],
                "block_embedding": emb["block_embedding"].tolist() if isinstance(emb["block_embedding"], np.ndarray) else emb["block_embedding"],
                "atom_embedding": emb["atom_embedding"].tolist() if isinstance(emb["atom_embedding"], np.ndarray) else emb["atom_embedding"],
                "block_id": emb["block_id"],
                "atom_id": emb["atom_id"]
            }
            embeddings_parquet.append(emb_copy)
        df = pd.DataFrame(embeddings_parquet)
        df.to_parquet(args.output_path, index=False)
        print(f"Saving processed data to {args.output_path} as parquet. Total of {len(embeddings)} items.")
    else:
        # Save as pickle file
        with open(args.output_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saving processed data to {args.output_path} as pickle. Total of {len(embeddings)} items.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default=None,
                        help='deprecated: pickled .ckpt files are no longer loadable, '
                             'use --model_config and --model_weights instead')
    parser.add_argument('--model_config', type=str, default=None, help='path of the model config to load')
    parser.add_argument('--model_weights', type=str, default=None, help='path of the model weights to load')
    parser.add_argument("--output_path", type=str, required=True, help='Path to save the output embeddings (supports .pkl or .parquet format)')
    parser.add_argument("--data_path", type=str, required=True, help='Path to the data file either in json, parquet, or pickle format')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to run inference on. 'auto' uses CUDA when available.")
    return parser.parse_args()


def cli():
    """Console script entry point for atomica-embeddings command.

    This function is called when running 'atomica-embeddings' from the command line.
    It parses command-line arguments and passes them to the main() function.

    Command-line usage:
        atomica-embeddings --model_config CONFIG.json --model_weights WEIGHTS.pt \\
            --data_path DATA.pkl --output_path OUTPUT.pkl
    """
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli()