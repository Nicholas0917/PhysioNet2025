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
export PRETRAIN_MODEL_FOLDER='/users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0724_2/Model'

echo "without pretrain weight"
echo "new model: ECGFeatureExtractor"
# echo "pre-train stage: update meta_net and classifier first (1e-4, 5 epoches)"
# echo "fine-tune stage: Differential Learning Rates for meta_net and classifier (lrx20)"

# # unzip ECG_signal.tar.gz to $TMP_SHARED
# start_time=$(date +%s)

# if command -v pigz &> /dev/null
# then
#     pigz -dc /mnt/scratch/wmqn2362/PhysioNet25/tmp/ECG_signal.tar.gz | tar -xf - -C $TMP_SHARED
# else
#     tar -xzf /mnt/scratch/wmqn2362/PhysioNet25/tmp/ECG_signal.tar.gz -C $TMP_SHARED
# fi

# end_time=$(date +%s)
# elapsed_time=$((end_time - start_time))
# echo "解压耗时: $elapsed_time 秒"

python train_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/train_folder/ \
    -m /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0724_2/Model \
    -v

python run_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -m /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0724_2/Model \
    -o /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0724_2/Output

python evaluate_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test_folder/ \
    -o /users/wmqn2362/PhysioNet2025/ECGFounder_add_Augmentation_0724_2/Output \

