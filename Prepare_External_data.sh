#!/bin/bash
#SBATCH --job-name=DataPreprocess         # Job name      
#SBATCH --time=2:00:00                   # Request 48 hours
#SBATCH --cpus-per-task=6                 # Number of cores per task
#SBATCH --mem-per-cpu=8G                 # Memory per CPU

# Request an email to be sent at the beginning and end of a job to the owner.
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wmqn2362@leeds.ac.uk

# Load your software to run
#SBATCH --export=NONE
module add miniforge/24.7.1

# Run the application, passing in the input and output filenames
conda activate PhysioNet25

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/cpsc_2018 \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/CSPC_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/cpsc_2018_extra \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/CSPC_extra_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/Chapman_Shaoxing_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/georgia \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/Georgia_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/ningbo \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/Ningbo_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/ptb \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/PTB_data.hdf5

# python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/ptb-xl \
#                                 --output_path /mnt/scratch/wmqn2362/PhysioNet25/PTBXL_data.hdf5

python Prepare_External_data.py --data_dir /mnt/scratch/wmqn2362/physionet.org/files/challenge-2021/1.0.3/training/st_petersburg_incart \
                                --output_path /mnt/scratch/wmqn2362/PhysioNet25/ST_Petersburg_data.hdf5

