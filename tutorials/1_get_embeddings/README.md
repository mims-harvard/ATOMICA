# Get embeddings from ATOMICA model

Required: A H100 or A100 GPU

## Activate your environment
If you are using a mamba environment, run:
mamba activate atomica-env

If you are using a virtual environment, run:
source atomica-env/bin/activate

## Download the model checkpoints from Hugging Face
Download the model checkpoints from Hugging Face to the checkpoints directory
You can use the hugging face CLI (setup instructructions for Hugging Face CLI [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli)) to download the pretrained model checkpoints:
```bash
hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints --include "ATOMICA_checkpoints/pretrain/**"
```

## Process the PDB files to extract the interaction interfaces 
python -m atomica.data.process_pdbs \
  --data_index_file data/example/example_inputs.csv \
  --out_path data/example/example_processed_data.parquet


## Embed the interfaces
python -m atomica.get_embeddings \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_embeddings.parquet