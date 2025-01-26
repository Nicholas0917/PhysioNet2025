#!/bin/bash
#SBATCH --job-name=PhysioNet_baseline     # Job name      
#SBATCH --time=48:00:00                   # Request 48 hours
#SBATCH --cpus-per-task=1                 # Number of cores per task
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

python train_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/train/ \
    -m /users/wmqn2362/PhysioNet2025/SEResNet/Model \
    -v

python run_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test/ \
    -m /users/wmqn2362/PhysioNet2025/SEResNet/Model \
    -o /users/wmqn2362/PhysioNet2025/SEResNet/Output

python evaluate_model.py -d /mnt/scratch/wmqn2362/PhysioNet25/test/ \
    -o /users/wmqn2362/PhysioNet2025/SEResNet/Output \