#!/bin/bash
#SBATCH -c 36 # Number of cores requested
#SBATCH -t 50:00:00
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
python train.py --gpu 0 --task pretrain_PPA --lr 0.0001 --final_lr 0.00002 --max_epoch 10000 --save_topk 10 --shuffle --model_type InteractNN --hidden_size 32 --n_layers 3 --radial_size 16 --n_channel 1 --n_rbf 16 --n_head 4 --k_neighbors 9  --valid_set ../GET/datasets/PPA/processed/split0/valid.pkl --train_set ../GET/datasets/PPA/processed/split0/train.pkl --save_dir ./pretrain/models/InteractNN --seed 2023 --run_name test_PPA_pretrain_interactnn_with_denoising_head --batch_size 4 --atom_level --use_wandb
#python3 -W ignore process_bindingmoad.py ./data/bindingmoad --make_split
#python3 extract_pockets_moad.py --dest './data/bindingmoad/pocket_10'
#python -m torch.distributed.launch train_ddp.py
#python3 optimization_fair.py
echo "Done."
