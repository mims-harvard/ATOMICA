import pickle
from openbabel import pybel
from tqdm import tqdm
import argparse
import json
import os
import sys

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..',
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from interaction_profiler.detection import InteractionProfile, SegmentType
from interaction_profiler.preparation import build_pybel_mols


def parse_args():
    parser = argparse.ArgumentParser(description='Profile interactions')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_path', type=str, help='Path to data file')
    parser.add_argument('--start_idx', type=int, help='Start index', default=0)
    parser.add_argument('--tmpfile_dir', type=str, help='Temporary file directory', default='./')
    parser.add_argument('--segment0_type', type=str, help='Segment 0 type', default='BINDING_SITE', choices=['BINDING_SITE', 'LIGAND'])
    parser.add_argument('--segment1_type', type=str, help='Segment 1 type', default='LIGAND', choices=['BINDING_SITE', 'LIGAND'])
    return parser.parse_args()

def main(args):
    ob_log_handler = pybel.ob.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)

    segment0_type = SegmentType[args.segment0_type]
    segment1_type = SegmentType[args.segment1_type]

    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)

    for data_idx in tqdm(range(args.start_idx, len(dataset)), total = len(dataset)-args.start_idx):
        item = dataset[data_idx]
        pybel_mols = build_pybel_mols(item['data'], tmpfile_dir=args.tmpfile_dir)
        ip = InteractionProfile(item, pybel_mols, segment0_type, segment1_type)
        summary = ip.summary
        json_string = json.dumps(summary)
        with open(args.output_path, 'a') as f:
            f.write(json_string + '\n')

if __name__ == '__main__':
    main(parse_args())