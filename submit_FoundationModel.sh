#!/bin/bash
#SBATCH --job-name=PhysioNet              # Job name      
#SBATCH --time=48:00:00                   # Request 48 hours
#SBATCH --cpus-per-task=6                 # Number of cores per task
#SBATCH --mem-per-cpu=8G                  # Memory per CPU
#SBATCH --partition=gpu                   # Partition (queue) name   
#SBATCH --gres=gpu:1                      # Request a single GPU

# Request an email to be sent at the beginning and end of a job to the owner.
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wmqn2362@leeds.ac.uk

# Load your software to run
#SBATCH --export=NONE
module add miniforge/24.7.1

# Run the application, passing in the input and output filenames
conda activate PhysioNet25

mkdir -p $TMP_SHARED
# cp -r /mnt/scratch/wmqn2362/PhysioNet25/tmp/* $TMP_SHARED
# cp /mnt/scratch/wmqn2362/PhysioNet25/tmp/ECG_combined_data.hdf5  $TMP_SHARED

export PYTHONUNBUFFERED=1

# Set the CACHE_FOLDER environment variable
export CACHE_FOLDER=$TMP_SHARED
export PRETRAIN_MODEL_FOLDER='/users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0731_2/Model'
export VISUALISATION_FOLDER='/users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0803_1/visualisation'

echo "ECGFeatureExtractor"
echo "Skip Pretrain"
echo "add DANN"
echo "Max lambda: 0.3"
echo "finetune_label_smoothing: 0.2"
echo "finetune_s: 1"
echo "finetune_margin: 0.8"
echo "pretrain_early_stop_patience: 8"

# move /mnt/scratch/wmqn2362/PhysioNet25/*.hdf5 to $TMP_SHARED
cp /mnt/scratch/wmqn2362/PhysioNet25/*.hdf5 $TMP_SHARED

python train_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/train_folder/ \
    -m /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0803_1/Model \
    -v

python run_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -m /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0803_1/Model \
    -o /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0803_1/Output

python evaluate_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -o /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0803_1/Output \
