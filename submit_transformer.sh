#!/bin/bash
#SBATCH --job-name=CTNTransformerPhysioNet  # Updated job name for transformer
#SBATCH --time=48:00:00                     
#SBATCH --cpus-per-task=6                   
#SBATCH --mem-per-cpu=8G                    
#SBATCH --partition=gpu                     
#SBATCH --gres=gpu:1                        
#SBATCH --output=transformer_job_%j.out     # Output file
#SBATCH --error=transformer_job_%j.err      # Error file

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=scabs@leeds.ac.uk       

#SBATCH --export=NONE
module load miniforge/24.7.1
source ~/.bashrc  
conda activate physionet

mkdir -p $TMP_SHARED

export PYTHONUNBUFFERED=1
export CACHE_FOLDER=$TMP_SHARED

# Updated paths for transformer model runs
export PRETRAIN_MODEL_FOLDER='/mnt/scratch/scabs/PhysioNet2025/Transformer_run/Model'

echo "Running CTN Transformer model"

# ---- Training ----
python train_model.py \
    -d /mnt/scratch/wmqn2362/PhysioNet25/train_folder/ \
    -m /mnt/scratch/scabs/PhysioNet2025/Transformer_run/Model \
    -v

# ---- Inference ----
python run_model.py \
    -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -m /mnt/scratch/scabs/PhysioNet2025/Transformer_run/Model \
    -o /mnt/scratch/scabs/PhysioNet2025/Transformer_run/Output

# ---- Evaluation ----
python evaluate_model.py \
    -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -o /mnt/scratch/scabs/PhysioNet2025/Transformer_run/Output
