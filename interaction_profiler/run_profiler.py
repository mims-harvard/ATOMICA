import subprocess
import os
import sys
import argparse
import pickle
import tempfile

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..',
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

parser = argparse.ArgumentParser(description='Profile interactions')
parser.add_argument('--data_path', type=str, help='Path to data file')
parser.add_argument('--output_path', type=str, help='Path to output file')
parser.add_argument('--tmpfile_dir', type=str, help='Temporary file directory', default='./')
parser.add_argument('--segment0_type', type=str, help='Segment 0 type', default='BINDING_SITE', choices=['BINDING_SITE', 'LIGAND'])
parser.add_argument('--segment1_type', type=str, help='Segment 1 type', default='LIGAND', choices=['BINDING_SITE', 'LIGAND'])

args = parser.parse_args()

# if os.path.exists(args.output_path):
#     raise ValueError(f"Output file {args.output_path} already exists!")

start_idx = 0
while True:
    # Run the script and wait for it to complete
    process = subprocess.Popen(['python', 'interaction_profiler/profile_interactions.py', f"--data_path={args.data_path}", 
                                f"--output_path={args.output_path}", f"--start_idx={start_idx}", f"--tmpfile_dir={args.tmpfile_dir}",
                                f"--segment0_type={args.segment0_type}", f"--segment1_type={args.segment1_type}"], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check if the script exited due to an error (non-zero exit code)
    if process.returncode != 0:
        start_idx += 1
        print(f"Detected crash, restarting at {start_idx}...")
        # Here, you might want to handle the error or log it
        print(stderr.decode())
        continue  # Restart the script
    break  # Exit loop if the script completed successfully

print("Finished!")



