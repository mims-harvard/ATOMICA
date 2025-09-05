from tqdm import tqdm
import pickle
import argparse
from joblib import Parallel, delayed, cpu_count

from .dataset import PDBDataset, blocks_to_data, data_to_blocks
from .converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def process_one(item, fragmentation_method):
    smiles = (item["id"].split("_")[1], item["id"].split("_")[3])
    try:
        list_of_blocks = data_to_blocks(item["data"])
        new_list_of_blocks = []
        for blocks, smi in zip(list_of_blocks, smiles):
            new_blocks = atom_blocks_to_frag_blocks(blocks, smiles=smi, fragmentation_method=fragmentation_method)
            new_list_of_blocks.append(new_blocks)
        new_data = blocks_to_data(*new_list_of_blocks)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None
    new_item = {"id": item["id"], "data": new_data, "affinity": item["affinity"]}
    return new_item


def main(args):
    csd_dataset = PDBDataset(args.data_file)
    result_list = pmap_multi(process_one, zip(csd_dataset.data), 
                            fragmentation_method=args.fragmentation_method, 
                            n_jobs=args.num_workers)

    processed_data = []
    for item in tqdm(result_list, desc="Processing complexes"):
        if item is None:
            continue
        processed_data.append(item)

    print(f"Saving processed data to {args.output}. Total of {len(processed_data)} items.")
    with open(args.output, "wb") as f:
        pickle.dump(processed_data, f)


def parse():
    parser = argparse.ArgumentParser(description='Tokenize processed CSD data')
    parser.add_argument('--data_file', type=str, required=True,
                        help='path to processed CSD data with no tokenization')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--fragmentation_method", type=str, default="PS_300")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse())