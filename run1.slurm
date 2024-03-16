#!/bin/bash
#SBATCH -c 16 # Number of cores requested
#SBATCH -t 100:00:00
#SBATCH -p kempner # Partition to submit to
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=50G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o ./run_logs/run_%j.out
#SBATCH -e ./run_logs/run_%j.err


echo "Loading environment..."
#source /n/app/miniconda3/4.10.3/etc/profile.d/conda.sh
source activate targetdiff
echo "Running model..."
#cp -r ../pretrainsbdd/data/crossdocked/crossdocked_pocket10 ./data/crossdocked/
python explain_infer.py --test_set /n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/data/test --task PLA --ckpt /n/holyscratch01/mzitnik_lab/afang/GET/datasets/LBA/split-by-sequence-identity-30/models/InteractNN/version_112/checkpoint/epoch144_step61190.ckpt --save_path ./output/results.jsonl --batch_size 32 --num_workers 4 --gpu 0
#python3 extract_pockets_moad.py --dest './data/bindingmoad/pocket_10'
#python -m torch.distributed.launch train_ddp.py
#python3 optimization_fair.py
echo "Done."
