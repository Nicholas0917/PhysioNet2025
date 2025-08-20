#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import copy
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial
import gc
import psutil
import requests
import json
import shutil # Import shutil for file operations

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
import torch.utils.model_zoo as model_zoo
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import from our modules
from dataset import *
from helper_code import *
from loss import *
from model import *
from utils import *

################################################################################
#
# Global configuration.
#
################################################################################

class Config:
    def __init__(self):
        # --- General & Path Settings ---
        self.model_name = os.getenv('MODEL_NAME', 'ECGFeatureExtractor')  # [ECGFeatureExtractor, ecgfounder, ResNet18, ResNet34, ResNet50]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Using a relative path is more robust as it relies on the WORKDIR set in the Dockerfile.
        self.download_folder = os.getenv('CACHE_FOLDER', './downloaded_data')
        # This is the folder where the script will create/process HDF5 files at runtime. 
        self.cache_folder = os.getenv('CACHE_FOLDER', '/tmp/wmqn2362/runtime_cache')

        self.pretrain_model_folder = os.getenv('PRETRAIN_MODEL_FOLDER', './Trained_models')  # This will be inside the container's WORKDIR
        self.pretrain_model_path = os.path.join(self.pretrain_model_folder, 'pretrain_model.pth')
        self.visualisation_folder = os.getenv('VISUALISATION_FOLDER', './tmp')
        self.num_preprocess_workers = os.cpu_count() // 4

        # --- Model Architecture ---
        self.model = {
            "use_pretrained": True,
            "dropout_rate": 0.3,
            "net1d_dropout_rate": 0.3,
            "meta_features": {
                "use_age": True,
                "use_sex": True,
                "use_signal_stats": False,
            }
        }
        
        # --- Pre-training Settings ---
        self.pretrain = {
            "num_epochs": 50,
            "learning_rate": 3e-5,
            "batch_size": 128,
            "early_stop_patience": 5,
            "loss": {
                "focal_gamma": 2,
                "margin": 0.1,
                "s": 30,
                "lmf_alpha": 0.98,
                "lmf_beta": 0.02,
                "label_smoothing": 0.0,
                "elr_lambda": 3.0,
                "elr_beta": 0.7,
                "elr_loss_weight": float(os.getenv('ELR_WEIGHT', 0.05))
            }
        }

        # --- Fine-tuning Settings ---
        self.finetune = {
            "num_epochs": 100,
            "learning_rate": 1e-6,
            "batch_size": 64,
            "early_stop_patience": 5,
            "is_train_encoder": bool(int(os.getenv('IS_TRAIN_ENCODER', 1))), # New parameter
            "loss": {
                "focal_gamma": 2,
                "margin": 0.8,
                "s": 1,
                "lmf_alpha": 0.98,
                "lmf_beta": 0.02,
                "label_smoothing": 0.2,
                "distill_lambda": float(os.getenv('DISTILL_LAMBDA', 0.05))
            }
        }
        
        # --- Domain-Adversarial Neural Network (DANN) Settings ---
        self.dann = {
            "num_domains": 8,
            "external_datasets": ['CODE15', 'CSPC', 'CSPC_extra', 'Chapman_Shaoxing', 'Georgia', 'Ningbo', 'PTB', 'ST_Petersburg'],
            "lambda": float(os.getenv('DANN_LAMBDA', 1.0)),  # Max weight for domain confusion loss
            "alpha": 10.0  # Steepness of the lambda scheduler
        }
        
        # --- Data Augmentation Settings ---
        self.augmentation = {
            "pos_sample_weight_multiplier": 1.0,
            "noise": {"use": True, "std": 0.03, "prob": 0.8},
            "scaling": {"use": True, "min": 0.5, "max": 2.0, "prob": 0.8},
            "flip": {"use": False, "prob": 0.2},
            "shift": {"use": True, "max_ratio": 0.8, "prob": 0.8},
            "drop": {"use": True, "max_prob": 0.02, "prob": 0.8},
            "power_noise": {"use": False, "amplitude": 0.03, "prob": 0.5},
            "cutout": {"use": True, "max_ratio": 0.2, "prob": 0.5},
            "lead_mixing": {"use": True, "lambda": 0.2, "prob": 0.8},
            "time_warp": {"use": True, "min_hz": 450, "max_hz": 550, "prob": 0.5},
            "baseline_wander": {"use": True, "min_freq": 0.05, "max_freq": 0.2, "amp_ratio": 0.2, "prob": 0.3}
        }

    def get_meta_feature_dim(self):
        dim = 0
        if self.model['meta_features']['use_age']: dim += 1
        if self.model['meta_features']['use_sex']: dim += 3
        if self.model['meta_features']['use_signal_stats']: dim += 2
        return dim
    
    def print_config(self):
        # This function remains the same.
        print(">>>>>>>>>>>>>>>>>>>>>>>>>Configuration:<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Model Name: {self.model_name}")
        print(">>>>>>>>>Pretraining Parameters:<<<<<<<<<<")
        print(f"Number of Epochs: {self.pretrain['num_epochs']}")
        print(f"Learning Rate: {self.pretrain['learning_rate']}")
        print(f"Batch Size: {self.pretrain['batch_size']}")
        print(f"Early Stop Patience: {self.pretrain['early_stop_patience']}")
        print(f"Pretrain ELR Lambda: {self.pretrain['loss']['elr_lambda']}")
        print(f"Pretrain ELR Beta: {self.pretrain['loss']['elr_beta']}")
        print(f"Pretrain ELR Loss Weight: {self.pretrain['loss']['elr_loss_weight']}")
        
        print(">>>>>>>>>Training Parameters:<<<<<<<<<<")
        print(f"Number of Epochs: {self.finetune['num_epochs']}")
        print(f"Learning Rate: {self.finetune['learning_rate']}")
        print(f"Dropout Rate: {self.model['dropout_rate']}")
        print(f"Net1D Dropout Rate: {self.model['net1d_dropout_rate']}")
        print(f"Batch Size: {self.finetune['batch_size']}")
        print(f"Early Stop Patience: {self.finetune['early_stop_patience']}")
        print(f"Is Train Encoder: {self.finetune['is_train_encoder']}") # New print
        print(f"Finetune Distill Lambda: {self.finetune['loss']['distill_lambda']}")
        
        print(">>>>>>>>>Meta Features:<<<<<<<<<<")
        print(f"Use Age: {self.model['meta_features']['use_age']}")
        print(f"Use Sex: {self.model['meta_features']['use_sex']}")
        print(f"Use Signal Stats: {self.model['meta_features']['use_signal_stats']}")
        print(f"Meta Feature Dimension: {self.get_meta_feature_dim()}")
        
        print(">>>>>>>>>DANN Parameters:<<<<<<<<<<")
        print(f"DANN Lambda: {self.dann['lambda']}")
        print(f"DANN Alpha: {self.dann['alpha']}")

        print(">>>>>>>>>Data Augmentation:<<<<<<<<<<")
        print(f"Use Noise Augmentation: {self.augmentation['noise']['use']}, Probability: {self.augmentation['noise']['prob']}")
        print(f"Use Scaling Augmentation: {self.augmentation['scaling']['use']}, Probability: {self.augmentation['scaling']['prob']}")
        print(f"Use Flip Augmentation: {self.augmentation['flip']['use']}, Probability: {self.augmentation['flip']['prob']}")
        print(f"Use Shift Augmentation: {self.augmentation['shift']['use']}, Max Ratio: {self.augmentation['shift']['max_ratio']}, Probability: {self.augmentation['shift']['prob']}")
        print(f"Use Drop Augmentation: {self.augmentation['drop']['use']}, Max Probability: {self.augmentation['drop']['max_prob']}, Probability: {self.augmentation['drop']['prob']}")
        print(f"Use 50Hz Power Noise: {self.augmentation['power_noise']['use']}, Probability: {self.augmentation['power_noise']['prob']}")
        print(f"Use Cutout Augmentation: {self.augmentation['cutout']['use']}, Max Ratio: {self.augmentation['cutout']['max_ratio']}, Probability: {self.augmentation['cutout']['prob']}")
        print(f"Use Lead Mixing Augmentation: {self.augmentation['lead_mixing']['use']}, Lambda: {self.augmentation['lead_mixing']['lambda']}, Probability: {self.augmentation['lead_mixing']['prob']}")
        print(f"Use Time Warping Augmentation: {self.augmentation['time_warp']['use']}, Freq Range: [{self.augmentation['time_warp']['min_hz']}, {self.augmentation['time_warp']['max_hz']}], Probability: {self.augmentation['time_warp']['prob']}")
        print(f"Use Baseline Wander Augmentation: {self.augmentation['baseline_wander']['use']}, Min Freq: {self.augmentation['baseline_wander']['min_freq']}, Max Freq: {self.augmentation['baseline_wander']['max_freq']}, Amp Ratio: {self.augmentation['baseline_wander']['amp_ratio']}, Probability: {self.augmentation['baseline_wander']['prob']}")

        print(">>>>>>>>>Loss Parameters:<<<<<<<<<<")
        print(f"Pretrain Focal Gamma: {self.pretrain['loss']['focal_gamma']}")
        print(f"Pretrain Margin: {self.pretrain['loss']['margin']}")
        print(f"Pretrain S: {self.pretrain['loss']['s']}")
        print(f"Pretrain LMF Alpha: {self.pretrain['loss']['lmf_alpha']}")
        print(f"Pretrain LMF Beta: {self.pretrain['loss']['lmf_beta']}")
        print(f"Pretrain Label Smoothing: {self.pretrain['loss']['label_smoothing']}")
        print(f"Finetune Focal Gamma: {self.finetune['loss']['focal_gamma']}")
        print(f"Finetune Margin: {self.finetune['loss']['margin']}")
        print(f"Finetune S: {self.finetune['loss']['s']}")
        print(f"Finetune LMF Alpha: {self.finetune['loss']['lmf_alpha']}")
        print(f"Finetune LMF Beta: {self.finetune['loss']['lmf_beta']}")
        print(f"Finetune Label Smoothing: {self.finetune['loss']['label_smoothing']}")
        print(f"Positive Sample Weight Multiplier: {self.augmentation['pos_sample_weight_multiplier']}")
        
        print(">>>>>>>>>Device:<<<<<<<<<<")
        print(f"Device: {self.device}")

config = Config()
config.print_config()

# Configure CUDA/cuDNN for stability
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# make cache folder if it does not exist
if not os.path.exists(config.cache_folder):
    os.makedirs(config.cache_folder)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose):
    """Train the model using the three-stage process"""
    # Create the runtime cache directory. This is the MOST IMPORTANT change.
    os.makedirs(config.cache_folder, exist_ok=True)
    if verbose:
        print(f"Runtime cache folder created at: {config.cache_folder}")
    
    start_total_time = time.time()

    ############################################################################
    # Stage 0: Data Loading and Preprocessing
    ############################################################################
    if verbose:
        print("Stage 0: Data Loading and Preprocessing...")
    stage0_start_time = time.time()
    records_relative = find_records(data_folder)
    records_full_path = [os.path.join(data_folder, r) for r in records_relative]
    
    # Define HDF5 file paths using the runtime cache folder for WRITING
    code15_hdf5_path = os.path.join(config.cache_folder, 'CODE15_data.hdf5')
    samitrop_hdf5_path = os.path.join(config.cache_folder, 'SaMiTrop_data.hdf5')
    ptbxl_hdf5_path = os.path.join(config.cache_folder, 'PTBXL_data.hdf5')
    
    # Initialize filters once
    highpass_filter_params, lowpass_filter_params, notch50_filter_params, notch60_filter_params = initialize_filters()

    cpu_num = os.cpu_count()
    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        results = list(executor.map(classify_record, records_full_path))
    
    code15_records_full_path = [r for r, src in results if src == 'CODE15']
    samitrop_records_full_path = [r for r, src in results if src == 'SaMiTrop']
    ptbxl_records_full_path = [r for r, src in results if src == 'PTBXL']
    
    del records_relative, results
    torch.cuda.empty_cache()
    gc.collect()

    def preprocess_and_write_to_hdf5(records_list, hdf5_path, dataset_name):
        if not os.path.exists(hdf5_path):
            start_time = time.time()
            print(f"Starting data preprocessing for {dataset_name} (creating HDF5 in {config.cache_folder})")

            if not records_list:
                print(f"No records for {dataset_name}. Skipping HDF5 creation.")
                return

            n_records = len(records_list)
            n_channels = 12
            n_samples = 5000
            meta_dim = config.get_meta_feature_dim()
            label_dim = 1
            chunk_n = os.cpu_count()

            dataset_to_domain_label = {
                'CODE15_data': 0,
                'CSPC_data': 1,
                'CSPC_extra_data': 2,
                'Chapman_Shaoxing_data': 3,
                'Georgia_data': 4,
                'Ningbo_data': 5,
                'PTB_data': 6,
                'ST_Petersburg_data': 7,
                'PTBXL_data': 8,
                'SaMiTrop_data': 9
            }
            domain_label_value = dataset_to_domain_label.get(dataset_name, -1)
            if domain_label_value == -1:
                print(f"Warning: Unknown dataset_name '{dataset_name}'. Domain label will be -1.")

            with h5py.File(hdf5_path, 'w') as hdf5_file:
                with ProcessPoolExecutor(max_workers=cpu_num) as executor:
                    signal_chunks = (min(chunk_n, n_records), n_samples, n_channels)
                    hdf5_file.create_dataset(
                        'signals',
                        shape=(n_records, n_samples, n_channels),
                        dtype='float32',
                        chunks=signal_chunks,
                        compression=None
                    )

                    hdf5_file.create_dataset(
                        'meta_features',
                        shape=(n_records, meta_dim),
                        dtype='float32',
                        compression=None
                    )

                    hdf5_file.create_dataset(
                        'labels',
                        shape=(n_records, label_dim),
                        dtype='float32',
                        compression=None
                    )
                    
                    hdf5_file.create_dataset(
                        'domain_labels',
                        shape=(n_records, label_dim),
                        dtype='float32',
                        compression=None
                    )
                    
                    pending = set()
                    for idx, record_path in enumerate(records_list):
                        fut = executor.submit(
                            data_preprocess,
                            record_path,
                            config,
                            highpass_filter_params,
                            lowpass_filter_params,
                            notch50_filter_params,
                            notch60_filter_params,
                            idx
                        )
                        pending.add(fut)

                        if len(pending) >= cpu_num:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for fut in done:
                                idx, signal, meta_features, label = fut.result()
                                hdf5_file['signals'][idx] = signal
                                hdf5_file['meta_features'][idx] = meta_features
                                hdf5_file['labels'][idx] = label
                                hdf5_file['domain_labels'][idx] = domain_label_value
                                del signal, meta_features, label

                    for fut in as_completed(pending):
                        idx, signal, meta_features, label = fut.result()
                        hdf5_file['signals'][idx] = signal
                        hdf5_file['meta_features'][idx] = meta_features
                        hdf5_file['labels'][idx] = label
                        hdf5_file['domain_labels'][idx] = domain_label_value
                        del signal, meta_features, label

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Created new {dataset_name} data file at {hdf5_path}")
            print(f"Data preprocessing and HDF5 creation for {dataset_name} took {elapsed_time:.2f} seconds.")
        else:
            print(f"Using existing {dataset_name} data file at {hdf5_path}")

    # The logic here is that if the HDF5 files for the training data don't exist in the writable cache, we create them there.
    # The external datasets are assumed to exist in the read-only download folder.
    if not os.path.exists(code15_hdf5_path):
        preprocess_and_write_to_hdf5(code15_records_full_path, code15_hdf5_path, 'CODE15_data')
    else:
        print(f"Skipping CODE15_data preprocessing as {code15_hdf5_path} already exists in cache.")

    if not os.path.exists(samitrop_hdf5_path):
        preprocess_and_write_to_hdf5(samitrop_records_full_path, samitrop_hdf5_path, 'SaMiTrop_data')
    else:
        print(f"Skipping SaMiTrop_data preprocessing as {samitrop_hdf5_path} already exists in cache.")

    if not os.path.exists(ptbxl_hdf5_path):
        preprocess_and_write_to_hdf5(ptbxl_records_full_path, ptbxl_hdf5_path, 'PTBXL_data')
    else:
        print(f"Skipping PTBXL_data preprocessing as {ptbxl_hdf5_path} already exists in cache.")

    del code15_records_full_path, samitrop_records_full_path, ptbxl_records_full_path

    stage0_end_time = time.time()
    if verbose:
        print(f"Stage 0 completed in {stage0_end_time - stage0_start_time:.2f} seconds.")
    
    ############################################################################
    # Stage 1: Pretrain Model
    ############################################################################
    if verbose:
        print("Stage 1: Pretrain Model...")
        
    # Load datasets. Pre-generated data is now read from the DOWNLOAD folder.
    # Data generated on-the-fly (from training folder) is read from the CACHE folder.
    pretrain_dataset = ECGDataset(dataset_name='CODE15', data_folder=config.cache_folder, is_training=True, config=config)
    external_datasets = []
    for dataset_name in config.dann['external_datasets']:
        if dataset_name == 'CODE15':
            data_folder_path = config.cache_folder
        else:
            data_folder_path = config.download_folder
        external_datasets.append(ECGDataset(dataset_name=dataset_name, data_folder=data_folder_path, is_training=True, config=config))

    stage1_start_time = time.time()
    # Initialize model and training components
    model = HybridModel(
        device=config.device,
        config=config
    )

    # First stage: update all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Print model parameters
    print_model_parameters(model, verbose)

    # Use create_binary_lmf_loss for pretraining
    pretrain_labels = [pretrain_dataset[i][1] for i in range(len(pretrain_dataset))]
    pretrain_neg_samples = pretrain_labels.count(0)
    pretrain_pos_samples = pretrain_labels.count(1)    
    pretrain_pos_samples_weighted = pretrain_pos_samples * config.augmentation['pos_sample_weight_multiplier']
    criterion = create_binary_lmf_loss(
        pos_samples=pretrain_pos_samples_weighted,
        neg_samples=pretrain_neg_samples,
        device=config.device,
        alpha=config.pretrain['loss']['lmf_alpha'],
        beta=config.pretrain['loss']['lmf_beta'],
        focal_gamma=config.pretrain['loss']['focal_gamma'],
        ldam_margin=config.pretrain['loss']['margin'],
        ldam_s=config.pretrain['loss']['s'],
        label_smoothing=config.pretrain['loss']['label_smoothing']
    )
    
    elr_criterion = ELRLoss(
        num_examples=len(pretrain_dataset),
        elr_lambda=config.pretrain['loss']['elr_lambda'],
        elr_beta=config.pretrain['loss']['elr_beta'],
        num_classes=1,
        device=config.device
    )
    model = pretrain_model(
        pretrain_dataset=pretrain_dataset,
        external_datasets=external_datasets,
        model=model,
        criterion=criterion,
        elr_criterion=elr_criterion, 
        num_epochs=config.pretrain['num_epochs'],
        batch_size=config.pretrain['batch_size'],
        early_stop_patience=config.pretrain['early_stop_patience'],
        device=config.device,
        pretrain_model_pth=os.path.join(model_folder, 'pretrain_model.pth'),
        verbose=verbose
    )
    
    # Close pretrain datasets
    pretrain_dataset.close()
    for ds in external_datasets:
        ds.close()
    del pretrain_dataset, external_datasets
    torch.cuda.empty_cache()
    gc.collect()

    stage1_end_time = time.time()
    if verbose:
        print(f"Stage 1 completed in {stage1_end_time - stage1_start_time:.2f} seconds.")

    ############################################################################
    # Stage 2: Evaluate pretrained model
    ############################################################################
    if verbose:
        print("Stage 2: Evaluate pretrained model...")
        stage2_start_time = time.time()
        
        # For evaluation, datasets are also loaded from their respective locations
        pretrain_eval_dataset = ECGDataset(dataset_name='CODE15', data_folder=config.cache_folder, is_training=False, config=config)
        samitrop_eval_dataset = ECGDataset(dataset_name='SaMiTrop', data_folder=config.cache_folder, is_training=False, config=config)
        ptbxl_eval_dataset = ECGDataset(dataset_name='PTBXL', data_folder=config.cache_folder, is_training=False, config=config)
        external_eval_datasets = []
        for dataset_name in config.dann['external_datasets']:
            if dataset_name == 'CODE15':
                data_folder_path = config.cache_folder
            else:
                data_folder_path = config.download_folder
            external_eval_datasets.append(ECGDataset(dataset_name=dataset_name, data_folder=data_folder_path, is_training=False, config=config))

        evaluate_model(model, pretrain_eval_dataset, samitrop_eval_dataset, ptbxl_eval_dataset, external_eval_datasets, verbose, 'pretrain')
        
        # Close evaluation datasets
        pretrain_eval_dataset.close()
        samitrop_eval_dataset.close()
        ptbxl_eval_dataset.close()
        for ds in external_eval_datasets:
            ds.close()
        del pretrain_eval_dataset, samitrop_eval_dataset, ptbxl_eval_dataset, external_eval_datasets
        torch.cuda.empty_cache()
        gc.collect()

        stage2_end_time = time.time()
        
        print(f"Stage 2 completed in {stage2_end_time - stage2_start_time:.2f} seconds.")

    ############################################################################
    # Stage 3: Finetune on target datasets
    ############################################################################
    
    samitrop_dataset = ECGDataset(dataset_name='SaMiTrop', data_folder=config.cache_folder, is_training=True, config=config)
    ptbxl_dataset = ECGDataset(dataset_name='PTBXL', data_folder=config.cache_folder, is_training=True, config=config)

    # --- Start of new logic for negative sampling ---
    num_positive_samitrop = len(samitrop_dataset)
    num_negative_needed = int(num_positive_samitrop * 49)
    negative_dataset_names = [name for name in config.dann['external_datasets'] if name != 'CODE15']
    negative_dataset_names.append('PTBXL')
    num_negative_datasets = len(negative_dataset_names)
    num_negative_per_dataset = num_negative_needed // num_negative_datasets

    print(f"SaMiTrop positive samples: {num_positive_samitrop}")
    print(f"Total negative samples needed: {num_negative_needed}")
    print(f"Negative datasets: {negative_dataset_names}")
    print(f"Negative samples per dataset: {num_negative_per_dataset}")

    sampled_negative_datasets = []
    actual_negative_datasets = [] 
    for ds_name in negative_dataset_names:
        if ds_name == 'PTBXL':
            current_dataset = ptbxl_dataset
        else:
            current_dataset = ECGDataset(dataset_name=ds_name, data_folder=config.download_folder, is_training=True, config=config)
            actual_negative_datasets.append(current_dataset)

        negative_indices = [i for i, (_, label, _, _) in enumerate(current_dataset) if label == 0]
        
        if len(negative_indices) > 0:
            num_samples_to_take = min(num_negative_per_dataset, len(negative_indices))
            sampled_indices = np.random.choice(negative_indices, num_samples_to_take, replace=False)
            sampled_negative_datasets.append(Subset(current_dataset, sampled_indices))
        else:
            print(f"Warning: No negative samples found in {ds_name} or dataset is empty.")

    finetune_dataset = [samitrop_dataset]
    finetune_dataset.extend(sampled_negative_datasets)
    finetune_dataset = torch.utils.data.ConcatDataset(finetune_dataset)
    # --- End of new logic for negative sampling ---

    if verbose:
        print("Stage 3: Finetune on target datasets...")
    stage3_start_time = time.time()
    
    print_model_parameters(model, verbose)

    finetune_labels = [finetune_dataset[i][1] for i in range(len(finetune_dataset))]
    finetune_neg_samples = finetune_labels.count(0)
    finetune_pos_samples = finetune_labels.count(1)

    finetune_pos_samples_weighted = finetune_pos_samples * config.augmentation['pos_sample_weight_multiplier']
    criterion = create_binary_lmf_loss(
        pos_samples=finetune_pos_samples_weighted,
        neg_samples=finetune_neg_samples,
        device=config.device,
        alpha=config.finetune['loss']['lmf_alpha'],
        beta=config.finetune['loss']['lmf_beta'],
        focal_gamma=config.finetune['loss']['focal_gamma'],
        ldam_margin=config.finetune['loss']['margin'],
        ldam_s=config.finetune['loss']['s'],
        label_smoothing=config.finetune['loss']['label_smoothing']
    )
    
    optimizers = []
    schedulers = []
    for _ in range(5):
        param_groups = []
        if config.finetune['is_train_encoder']:
            param_groups.append({'params': list(model.encoder.parameters()), 'lr': config.finetune['learning_rate'], 'name': 'encoder'})
        param_groups.append({'params': model.classifier.parameters(), 'lr': config.finetune['learning_rate'] * 5, 'name': 'classifier'})
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.finetune['num_epochs']
        )
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    kf = StratifiedKFold(n_splits=5)
    
    finetuned_models = finetune_model(
        model=model,
        finetune_dataset=finetune_dataset,
        model_folder=model_folder,
        verbose=verbose,
        criterion=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        kf=kf
    )
    
    samitrop_dataset.close()
    ptbxl_dataset.close()
    for ds in actual_negative_datasets:
        ds.close()
    del samitrop_dataset, ptbxl_dataset, actual_negative_datasets, finetune_dataset
    torch.cuda.empty_cache()
    gc.collect()

    stage3_end_time = time.time()
    if verbose:
        print(f"Stage 3 completed in {stage3_end_time - stage3_start_time:.2f} seconds.")

    ############################################################################
    # Stage 4: Evaluate finetuned model
    ############################################################################
    if verbose:
        print("Stage 4: Evaluate finetuned model...")
    stage4_start_time = time.time()
    if verbose:
        pretrain_eval_dataset = ECGDataset(dataset_name='CODE15', data_folder=config.cache_folder, is_training=False, config=config)
        samitrop_eval_dataset = ECGDataset(dataset_name='SaMiTrop', data_folder=config.cache_folder, is_training=False, config=config)
        ptbxl_eval_dataset = ECGDataset(dataset_name='PTBXL', data_folder=config.cache_folder, is_training=False, config=config)
        external_eval_datasets = []
        for dataset_name in config.dann['external_datasets']:
            if dataset_name == 'CODE15':
                data_folder_path = config.cache_folder
            else:
                data_folder_path = config.download_folder
            external_eval_datasets.append(ECGDataset(dataset_name=dataset_name, data_folder=data_folder_path, is_training=False, config=config))
        evaluate_model(finetuned_models, pretrain_eval_dataset, samitrop_eval_dataset, ptbxl_eval_dataset, external_eval_datasets, verbose, 'finetune')

    stage4_end_time = time.time()
    if verbose:
        print(f"Stage 4 completed in {stage4_end_time - stage4_start_time:.2f} seconds.")

    end_total_time = time.time()
    if verbose:
        print(f"Total training process completed in {end_total_time - start_total_time:.2f} seconds.")
        print('Done.')
        print()

def pretrain_model(pretrain_dataset, external_datasets, model, criterion, elr_criterion, # Add elr_criterion
                  num_epochs, batch_size, early_stop_patience, device,
                  pretrain_model_pth, verbose):
    """Core pretraining logic with training loop and model saving, incorporating DANN"""
    # torch.autograd.set_detect_anomaly(True)
    os.makedirs(os.path.dirname(pretrain_model_pth), exist_ok=True)
    
    if os.path.exists(config.pretrain_model_path): # Now config.pretrain_model_path is the full file path
        if verbose:
            print(f"Found existing pretrained model at {config.pretrain_model_path}. Loading model and skipping pretraining.")
        model.load_state_dict(torch.load(config.pretrain_model_path, map_location=device))
        # The model is already loaded from config.pretrain_model_path,
        # so saving it to pretrain_model_pth (which is the same path if pretrain_model_path is correctly set)
        # is redundant but harmless. Keep it for consistency with original logic.
        torch.save(model.state_dict(), pretrain_model_pth)
        if verbose:
            print(f"Loaded pretrained model saved to {pretrain_model_pth}.")
        return model

    if verbose:
        print(f'Starting pretraining on CODE-15% ({len(pretrain_dataset)} records) and External ({sum(len(ds) for ds in external_datasets)} records) for domain adaptation.')

    # Create data loaders for all datasets
    # CODE-15% dataset (source domain for DANN and task classification)
    code15_train_size = int(0.8 * len(pretrain_dataset))
    code15_val_size = len(pretrain_dataset) - code15_train_size
    code15_train_dataset, code15_val_dataset = torch.utils.data.random_split(pretrain_dataset, [code15_train_size, code15_val_size])

    code15_train_weights = make_weights_for_balanced_classes(code15_train_dataset)
    code15_train_sampler = WeightedRandomSampler(code15_train_weights, len(code15_train_weights))
    
    # DataLoader for Chagas classification (full batch_size)
    code15_chagas_loader = DataLoader(code15_train_dataset,
                                      batch_size=batch_size,
                                      sampler=code15_train_sampler,
                                      num_workers=config.num_preprocess_workers)
    code15_val_loader = DataLoader(code15_val_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=config.num_preprocess_workers)

    # External datasets (source domains for DANN)
    external_train_loaders = []
    external_val_loaders = []
    for ext_ds in external_datasets:
        print(ext_ds.dataset_name, len(ext_ds), "records")
        ext_train_size = int(0.8 * len(ext_ds))
        ext_val_size = len(ext_ds) - ext_train_size
        ext_train_dataset, ext_val_dataset = torch.utils.data.random_split(ext_ds, [ext_train_size, ext_val_size])
        
        ext_train_loader = DataLoader(ext_train_dataset,
                                      batch_size=batch_size // config.dann['num_domains'],
                                      shuffle=True,
                                      num_workers=config.num_preprocess_workers)
        ext_val_loader = DataLoader(ext_val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=config.num_preprocess_workers)
        external_train_loaders.append(ext_train_loader)
        external_val_loaders.append(ext_val_loader)

    # Define optimizers for different parts of the model
    # Optimizer for feature extractor (model.encoder) and task classifier (model.classifier)
    optimizer_task = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=config.pretrain['learning_rate'],
        weight_decay=2e-4
    )
    
    # Optimizer for domain classifier (model.domain_classifier)
    optimizer_domain_classifier = torch.optim.AdamW(
        model.domain_classifier.parameters(),
        lr=config.pretrain['learning_rate'] * 0.1, # Can be different from task optimizer LR
        weight_decay=2e-4
    )

    # Optimizer for encoder (model.encoder) for confusion
    optimizer_encoder_confusion = torch.optim.AdamW(
        model.encoder.parameters(), # Only encoder parameters
        lr=config.pretrain['learning_rate'],
        weight_decay=2e-4
    )

    # Schedulers for all optimizers
    warmup_epochs = int(num_epochs * 0.1)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + (epoch / warmup_epochs) * 0.9
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler_task = torch.optim.lr_scheduler.LambdaLR(optimizer_task, lr_lambda)
    scheduler_domain_classifier = torch.optim.lr_scheduler.LambdaLR(optimizer_domain_classifier, lr_lambda)
    scheduler_encoder_confusion = torch.optim.lr_scheduler.LambdaLR(optimizer_encoder_confusion, lr_lambda)

    # Loss for domain classification
    domain_criterion = nn.CrossEntropyLoss()
    # Loss for confusion (same as domain criterion, but we will maximize it)
    confusion_criterion = ConfusionLoss()

    best_auprc = 0.0 # This will track AUPRC for Chagas classification on CODE-15% validation set
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    # Define the epoch from which early stopping should start
    start_early_stopping_epoch = 15 # User wants to start early stopping from epoch 15

    def calculate_lambda(epoch, num_epochs, max_lambda=config.dann['lambda'], alpha=config.dann['alpha']):
        progress = epoch / num_epochs
        p = 2.0 / (1.0 + math.exp(-alpha * progress)) - 1.0
        return max_lambda * p
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_chagas_loss = 0.0
        total_elr_loss = 0.0
        total_combined_chagas_loss = 0.0
        total_domain_loss = 0.0
        total_confusion_loss = 0.0
        train_targets = []
        train_outputs = []
        train_domain_targets = []
        train_domain_outputs = []

        # For Chagas logit printing (training)
        train_pos_logits_sum = 0.0
        train_neg_logits_sum = 0.0
        train_pos_count = 0
        train_neg_count = 0

        # Iterators for all data loaders
        code15_chagas_iter = iter(code15_chagas_loader)
        external_iters = [iter(loader) for loader in external_train_loaders] # Use external_train_loaders directly

        # Determine the maximum number of batches to iterate over for domain adaptation
        max_batches = len(code15_chagas_loader)
        
        for i in range(max_batches):
            # ------------------------------------------------------------------
            # --- Phase 1: Task Training (Update Encoder and Label Predictor) ---
            # ------------------------------------------------------------------
            model.train()
            # Set requires_grad for task-related parameters
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.domain_classifier.parameters():
                param.requires_grad = False # Fix domain predictor

            optimizer_task.zero_grad()
            try:
                code15_features_chagas, code15_label_chagas, code15_domain_label, index = next(code15_chagas_iter)
            except StopIteration:
                code15_chagas_iter = iter(code15_chagas_loader)
                code15_features_chagas, code15_label_chagas, code15_domain_label, index = next(code15_chagas_iter)
            
            signal_chagas, meta_chagas = code15_features_chagas
            signal_chagas = signal_chagas.to(device)
            meta_chagas = meta_chagas.to(device)
            code15_label_chagas = code15_label_chagas.to(device).view(-1, 1)

            task_output_chagas, _, _ = model(signal_chagas, meta_chagas) 
            chagas_loss = criterion(task_output_chagas, code15_label_chagas)
            
            # Add ELR loss with warmup
            elr_loss = elr_criterion(index, task_output_chagas, code15_label_chagas)
            if epoch >= 5: # Apply ELR loss after 5 epochs
                chagas_loss_with_elr = chagas_loss + config.pretrain['loss']['elr_loss_weight'] * elr_loss
            else:
                chagas_loss_with_elr = chagas_loss

            chagas_loss_with_elr.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_task.step()
            
            total_chagas_loss += chagas_loss.item()
            if epoch >= 5: # Accumulate ELR loss only after warmup
                total_elr_loss += elr_loss.item()
            total_combined_chagas_loss += chagas_loss_with_elr.item()
            train_targets.extend(code15_label_chagas.cpu().numpy())
            train_outputs.extend(torch.sigmoid(task_output_chagas).detach().cpu().numpy())

            # Accumulate positive/negative logit stats for training
            if verbose:
                pos_mask_train = code15_label_chagas == 1
                neg_mask_train = ~pos_mask_train
                
                if pos_mask_train.any():
                    train_pos_logits_sum += task_output_chagas[pos_mask_train].sum().item()
                    train_pos_count += pos_mask_train.sum().item()
                if neg_mask_train.any():
                    train_neg_logits_sum += task_output_chagas[neg_mask_train].sum().item()
                    train_neg_count += neg_mask_train.sum().item()

            # ------------------------------------------------------------------
            # --- Phase 2: Domain Classifier Training (Update Domain Classifier) ---
            # ------------------------------------------------------------------
            model.train()
            # Set requires_grad for domain classifier parameters
            for param in model.encoder.parameters():
                param.requires_grad = False # Fix encoder
            for param in model.classifier.parameters():
                param.requires_grad = False # Fix label predictor
            for param in model.domain_classifier.parameters():
                param.requires_grad = True

            optimizer_domain_classifier.zero_grad()
            # Prepare combined data for Domain Adversarial and Confusion Training
            all_signals_domain = []
            all_metas_domain = []
            all_domain_labels_combined = []

            # Add all DANN datasets for domain tasks
            for j, ext_iter in enumerate(external_iters): # Iterate over external_iters
                try:
                    ext_features, ext_label, ext_domain_label, ext_idx = next(ext_iter)
                except StopIteration:
                    external_iters[j] = iter(external_train_loaders[j]) # Use external_train_loaders here
                    ext_features, ext_label, ext_domain_label, ext_idx = next(external_iters[j])
                
                signal_ext, meta_ext = ext_features
                all_signals_domain.append(signal_ext)
                all_metas_domain.append(meta_ext)
                all_domain_labels_combined.append(ext_domain_label.to(device))
            
            combined_domain_signal = torch.cat(all_signals_domain, 0).to(device)
            combined_domain_meta = torch.cat(all_metas_domain, 0).to(device)
            combined_domain_target = torch.cat(all_domain_labels_combined, 0).to(device)
            # Add .squeeze() to convert (N, 1) to (N)
            combined_domain_target = combined_domain_target.squeeze()
            # Convert target to long type for CrossEntropyLoss
            combined_domain_target = combined_domain_target.long()

            # Use detached features to prevent gradient flow back to encoder
            _, domain_output_combined, _ = model(combined_domain_signal, combined_domain_meta)
            domain_loss = domain_criterion(domain_output_combined, combined_domain_target)
            
            domain_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_domain_classifier.step()
            total_domain_loss += domain_loss.item()
            train_domain_targets.extend(combined_domain_target.cpu().numpy())
            train_domain_outputs.extend(torch.argmax(domain_output_combined, dim=1).detach().cpu().numpy())

            if epoch >= 10: # Warmup for 5 epochs without DANN
                # ------------------------------------------------------------------
                # --- Phase 3: Confusion Training (Update Encoder) ---
                # ------------------------------------------------------------------
                model.train()
                # Set requires_grad for encoder parameters
                for param in model.domain_classifier.parameters():
                    param.requires_grad = False # Fix domain predictor
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.classifier.parameters():
                    param.requires_grad = False # Fix label predictor

                optimizer_encoder_confusion.zero_grad()
                # Re-run forward pass to get features with gradients enabled for encoder
                _, domain_output_confusion, _ = model(combined_domain_signal, combined_domain_meta)
                
                # Maximize domain classifier error: use negative of domain loss
                # Dynamically calculate dann_lambda
                dann_lambda = calculate_lambda(epoch - 10, num_epochs, max_lambda=config.dann['lambda'], alpha=config.dann['alpha'])
                confusion_loss = dann_lambda * confusion_criterion(domain_output_confusion, combined_domain_target)
                
                confusion_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_encoder_confusion.step()
                total_confusion_loss += confusion_loss.item()
        
        total_chagas_loss /= len(code15_chagas_loader) # Average over code15 batches
        total_elr_loss /= len(code15_chagas_loader) # Average over code15 batches
        total_combined_chagas_loss /= len(code15_chagas_loader) # Average over code15 batches
        total_domain_loss /= max_batches # Average over DANN batches
        total_confusion_loss /= max_batches # Average over DANN batches

        # Calculate epoch averages for training Chagas logits
        if verbose:
            epoch_train_pos_logit = train_pos_logits_sum / train_pos_count if train_pos_count > 0 else 0.0
            epoch_train_neg_logit = train_neg_logits_sum / train_neg_count if train_neg_count > 0 else 0.0

        # ------------------------------------------------------------------
        # --- Validation on CODE-15% for Chagas classification ---
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []
        
        # For Chagas logit printing (validation)
        val_pos_logits_sum = 0.0
        val_neg_logits_sum = 0.0
        val_pos_count = 0
        val_neg_count = 0
        
        with torch.no_grad():
            for features, label, domain_label, idx in code15_val_loader:
                signal, meta_features = features
                signal = signal.to(device)
                meta_features = meta_features.to(device)
                label = label.to(device)

                task_output, _, _ = model(signal, meta_features)
                label_reshaped = label.view(-1, 1)
                loss = criterion(task_output, label_reshaped)
                
                val_loss += loss.item()
                val_targets.extend(label.cpu().numpy())
                val_outputs.extend(torch.sigmoid(task_output).detach().cpu().numpy())

                # Accumulate positive/negative logit stats for validation
                if verbose:
                    pos_mask_val = label == 1
                    neg_mask_val = ~pos_mask_val
                    if pos_mask_val.any():
                        val_pos_logits_sum += task_output[pos_mask_val].sum().item()
                        val_pos_count += pos_mask_val.sum().item()
                    if neg_mask_val.any():
                        val_neg_logits_sum += task_output[neg_mask_val].sum().item()
                        val_neg_count += neg_mask_val.sum().item()

        val_loss /= len(code15_val_loader)
        
        val_outputs_arr = np.array(val_outputs)
        val_targets_arr = np.array(val_targets)

        if verbose:
            # Calculate validation epoch averages
            val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
            val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
            
        if len(np.unique(val_targets_arr)) < 2:
            print("WARNING: Only one class present in validation targets")
            val_auroc = 0.5
        else:
            val_auroc = roc_auc_score(val_targets_arr, val_outputs_arr)
            
        val_auprc = average_precision_score(val_targets_arr, val_outputs_arr)
        val_accuracy = accuracy_score(val_targets_arr, np.round(val_outputs_arr))
        val_f1 = f1_score(val_targets_arr, np.round(val_outputs_arr))

        epoch_duration = time.time() - epoch_start_time

        print(f'Epoch {epoch + 1}/{num_epochs}, Chagas Loss: {total_chagas_loss:.4f}, ELR Loss: {total_elr_loss:.4f}, Combined Chagas Loss: {total_combined_chagas_loss:.4f}, Domain Loss: {total_domain_loss:.4f}, Confusion Loss: {total_confusion_loss:.4f}, Valid Loss (Chagas): {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
        print(f'Valid AUROC (Chagas): {val_auroc:.4f}, Valid AUPRC (Chagas): {val_auprc:.4f}, Valid Accuracy (Chagas): {val_accuracy:.4f}, Valid F1 (Chagas): {val_f1:.4f}')
        
        if verbose:
            print(f"Train Averages (Chagas) - Positive Logit: {epoch_train_pos_logit:.4f}, Negative Logit: {epoch_train_neg_logit:.4f}")
            # Calculate validation epoch averages
            val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
            val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
            
            print(f"Valid Averages (Chagas) - Positive Logit: {val_epoch_pos_logit:.4f}, Negative Logit: {val_epoch_neg_logit:.4f}")

        # ------------------------------------------------------------------
        # --- Domain Accuracy on all datasets ---
        # ------------------------------------------------------------------
        for j, ext_val_loader in enumerate(external_val_loaders):
            ext_val_domain_targets = []
            ext_val_domain_outputs = []
            with torch.no_grad():
                for features, label, domain_label, idx in ext_val_loader: # label is not used for external datasets in this context
                    signal, meta_features = features
                    signal = signal.to(device)
                    meta_features = meta_features.to(device)
                    
                    _, domain_output, _ = model(signal, meta_features)
                    
                    domain_label_ext_val = domain_label.to(device) # Directly use domain_label from loader
                    ext_val_domain_targets.extend(domain_label_ext_val.cpu().numpy())
                    ext_val_domain_outputs.extend(torch.argmax(domain_output, dim=1).detach().cpu().numpy())
            
            ext_domain_accuracy = accuracy_score(ext_val_domain_targets, ext_val_domain_outputs)
            print(f'Valid Domain Accuracy ({config.dann["external_datasets"][j]}): {ext_domain_accuracy:.4f}')
        print('\n')

        scheduler_task.step()
        scheduler_domain_classifier.step()
        if epoch >= 10: 
            scheduler_encoder_confusion.step()

        if epoch + 1 >= start_early_stopping_epoch:
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping: Valid AUPRC not improved for {early_stop_patience} epochs")
                    break

    if best_model_state:
        torch.save(best_model_state, pretrain_model_pth)
        model.load_state_dict(best_model_state)
        if verbose:
            print(f'Pretraining completed and best model from epoch {best_epoch + 1} saved to {pretrain_model_pth}')
    else:
        torch.save(model.state_dict(), pretrain_model_pth)
        if verbose:
            print(f'Pretraining completed and current model saved to {pretrain_model_pth} (no improvement).')
    
    return model

def finetune_model(model, finetune_dataset, model_folder, verbose, criterion, optimizers,
                  schedulers, kf):
    """Finetune the model on target datasets
    
    Args:
        model: Pretrained model with frozen layers
        finetune_dataset: ECGDataset for finetuning
        model_folder: Path to save model
        verbose: Whether to print progress
        criterion: Loss function
        optimizers: List of optimizers (one per fold)
        schedulers: List of schedulers (one per fold)
        scaler: Gradient scaler for mixed precision
        autocast: Autocast context manager
        kf: StratifiedKFold instance
    """
    # torch.autograd.set_detect_anomaly(True)
    
    # Check if all fold models already exist
    all_folds_exist = all(
        os.path.exists(os.path.join(config.pretrain_model_folder, f'model_fold{fold_num}.pth'))
        for fold_num in range(1, 6)
    )

    if all_folds_exist:
        if verbose:
            print(f"All 5 fold models found in {config.pretrain_model_folder}. Skipping finetuning and loading models.")
            
        loaded_models = load_model(config.pretrain_model_folder, verbose)

        os.makedirs(model_folder, exist_ok=True)
        
        for i, model in enumerate(loaded_models):
            fold_num = i + 1
            save_model(model_folder, model.state_dict(), config, fold=fold_num)
            if verbose:
                print(f"Saved loaded model for fold {fold_num} to {model_folder}")
        
        return loaded_models

    # Save initial model state
    initial_state = copy.deepcopy(model.state_dict())
    
    if verbose:
        print('Training the model on the fine-tune data...')
        print("Fine-tune Datastes Size: ", len(finetune_dataset))
    
    # Training loop with stratified k-fold
    X = [finetune_dataset[i][0] for i in range(len(finetune_dataset))]
    labels = [finetune_dataset[i][1] for i in range(len(finetune_dataset))]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        if verbose:
            print(f'Fold {fold + 1}')
        
        # Reset model to initial state for each fold
        model.load_state_dict(torch.load(os.path.join(model_folder, 'pretrain_model.pth')))
        
        # Create a teacher model (a copy of the encoder) and freeze its parameters
        encoder_teacher = copy.deepcopy(model.encoder)
        for param in encoder_teacher.parameters():
            param.requires_grad = False
        encoder_teacher.eval()
        encoder_teacher.to(config.device)

        # Ensure encoder and classifier parameters are trainable for finetuning
        for param in model.encoder.parameters():
            param.requires_grad = config.finetune['is_train_encoder'] # Control based on new parameter
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.domain_classifier.parameters():
            param.requires_grad = False

        train_subset = Subset(finetune_dataset, train_idx)
        val_subset = Subset(finetune_dataset, val_idx)

        # Create weighted sampler for imbalanced data
        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(train_subset, 
                                batch_size=config.finetune['batch_size'], 
                                sampler=train_sampler,
                                num_workers=config.num_preprocess_workers,
                                drop_last=True)
        val_loader = DataLoader(val_subset, 
                              batch_size=config.finetune['batch_size'], 
                              shuffle=False,
                              num_workers=config.num_preprocess_workers,
                              drop_last=True)

        best_val_loss = float('inf')
        best_epoch = 0
        start_time = time.time()

        for epoch in range(config.finetune['num_epochs']):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            train_chagas_loss = 0.0
            train_distill_loss = 0.0
            train_targets = []
            train_outputs = []
            if verbose:
                pos_logits_sum = 0.0
                neg_logits_sum = 0.0
                pos_count = 0
                neg_count = 0
            
            for i, (features, label, domain_label, idx) in enumerate(train_loader):
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                label = label.to(config.device)

                optimizers[fold].zero_grad()
                
                # Student model forward pass
                task_output, _, features_student = model(signal, meta_features)
                
                # Teacher model forward pass (no_grad)
                with torch.no_grad():
                    features_teacher = encoder_teacher(signal, meta_features)
                
                # Reshape label to match output shape [batch_size, 1]
                label_reshaped = label.view(-1, 1)
                
                # Chagas loss
                chagas_loss = criterion(task_output, label_reshaped)
                
                # Feature distillation loss (only if encoder is being trained)
                if config.finetune['is_train_encoder']:
                    distill_loss = F.mse_loss(features_student, features_teacher)
                    loss = chagas_loss + config.finetune['loss']['distill_lambda'] * distill_loss
                    train_distill_loss += distill_loss.item()
                else:
                    loss = chagas_loss
                    train_distill_loss = 0.0 # No distillation loss if encoder is frozen
                
                loss.backward()
                check_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizers[fold].step()

                train_loss += loss.item()
                train_chagas_loss += chagas_loss.item()
                train_targets.extend(label.cpu().numpy())
                train_outputs.extend(torch.sigmoid(task_output).detach().cpu().numpy())

                if verbose:
                    # Calculate positive/negative logit stats
                    pos_mask = label == 1
                    neg_mask = ~pos_mask
                    if pos_mask.any():
                        pos_logits = task_output[pos_mask].sum().item()
                    else:
                        pos_logits = 0.0
                    if neg_mask.any():
                        neg_logits = task_output[neg_mask].sum().item()
                    else:
                        neg_logits = 0.0
                    
                    # Accumulate stats for epoch average
                    if pos_mask.any():
                        pos_logits_sum += pos_logits
                        pos_count += pos_mask.sum().item()
                    if neg_mask.any():
                        neg_logits_sum += neg_logits
                        neg_count += neg_mask.sum().item()

            train_loss /= len(train_loader)
            train_chagas_loss /= len(train_loader)
            train_distill_loss /= len(train_loader)

            schedulers[fold].step()

            # Calculate training metrics
            train_auroc = roc_auc_score(train_targets, train_outputs)
            train_auprc = average_precision_score(train_targets, train_outputs)
            train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
            train_f1 = f1_score(train_targets, np.round(train_outputs))

            # Validation
            model.eval()
            val_loss = 0.0
            val_chagas_loss = 0.0
            val_targets = []
            val_outputs = []
            if verbose:
                val_pos_logits_sum = 0.0
                val_neg_logits_sum = 0.0
                val_pos_count = 0
                val_neg_count = 0

            with torch.no_grad():
                for i, (features, label, domain_label, idx) in enumerate(val_loader):
                    signal, meta_features = features
                    signal = signal.to(config.device)
                    meta_features = meta_features.to(config.device)
                    label = label.to(config.device)

                    task_output, _, _ = model(signal, meta_features)
                    # Reshape label to match output shape [batch_size, 1]
                    label_reshaped = label.view(-1, 1)
                    loss = criterion(task_output, label_reshaped)
                    
                    val_loss += loss.item()
                    val_chagas_loss += loss.item()
                    val_targets.extend(label.cpu().numpy())
                    val_outputs.extend(torch.sigmoid(task_output).detach().cpu().numpy())
                    if verbose:
                        # Calculate positive/negative logit stats for validation
                        pos_mask = label == 1
                        neg_mask = ~pos_mask
                        if pos_mask.any():
                            pos_logits = task_output[pos_mask].sum().item()
                            val_pos_logits_sum += pos_logits
                            val_pos_count += pos_mask.sum().item()
                        if neg_mask.any():
                            neg_logits = task_output[neg_mask].sum().item()
                            val_neg_logits_sum += neg_logits
                            val_neg_count += neg_mask.sum().item()

            val_loss /= len(val_loader)
            val_chagas_loss /= len(val_loader)
            
            val_auroc = roc_auc_score(val_targets, val_outputs)
            val_auprc = average_precision_score(val_targets, val_outputs)
            val_accuracy = accuracy_score(val_targets, np.round(val_outputs))
            val_f1 = f1_score(val_targets, np.round(val_outputs))

            epoch_duration = time.time() - epoch_start_time
            
            if verbose:
                print(f'Fold {fold + 1}, Epoch {epoch + 1}/{config.finetune["num_epochs"]}:')
                print(f'  Train Loss: {train_loss:.4f}, Chagas Loss: {train_chagas_loss:.4f}, Distill Loss: {train_distill_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}')
                print(f'  Valid Loss: {val_loss:.4f}, Chagas Loss: {val_chagas_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, Time: {epoch_duration:.2f}s')
                # Calculate epoch averages
                epoch_pos_logit = pos_logits_sum / pos_count if pos_count > 0 else 0.0
                epoch_neg_logit = neg_logits_sum / neg_count if neg_count > 0 else 0.0
                val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
                val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
                
                print(f"  Train Logits: Pos {epoch_pos_logit:.4f}, Neg {epoch_neg_logit:.4f}")
                print(f"  Valid Logits: Pos {val_epoch_pos_logit:.4f}, Neg {val_epoch_neg_logit:.4f}")

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.finetune['early_stop_patience']:
                    if verbose:
                        print(f"Early stopping: Valid Loss not improved for {config.finetune['early_stop_patience']} epochs")
                    break

            # del train_targets, train_outputs, val_targets, val_outputs
            # torch.cuda.empty_cache()
            # gc.collect()
        
        end_time = time.time()
        if verbose:
            print(f'Fold {fold + 1} finished. Best Valid Loss: {best_val_loss:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

        # Save finetuned model
        os.makedirs(model_folder, exist_ok=True)
        save_model(model_folder, best_model, config, fold=fold+1)
        if verbose:
            print(f'Finetuning completed for fold {fold+1}')
    
    # After all folds are trained (or skipped), load all models and return them
    finetuned_models = load_model(model_folder, verbose)
    return finetuned_models

def evaluate_model(models, code15_dataset, samitrop_dataset, ptbxl_dataset, external_datasets, verbose, stage_name):
    """
    Evaluate model performance on both pretrain and finetune datasets
    
    Args:
        models: Single model or list of models
        code15_dataset, samitrop_dataset, ptbxl_dataset: Evaluation datasets
        external_datasets: List of external datasets
        verbose: Whether to print detailed information
        stage_name: Stage identifier for file naming
    """
    # Ensure models is always a list for uniform processing
    if not isinstance(models, list):
        models = [models]
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    os.makedirs(config.visualisation_folder, exist_ok=True)
    
    def collect_features_and_predictions(dataset, num_samples=None):
        """Collect encoder features and predictions from all models for a dataset"""
        # Prepare data loader
        if num_samples is not None:
            indices = np.random.choice(len(dataset), min(len(dataset), num_samples), replace=False)
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=config.finetune['batch_size'], shuffle=False, 
                              num_workers=config.num_preprocess_workers)
        else:
            loader = DataLoader(dataset, batch_size=config.finetune['batch_size'], shuffle=False,
                              num_workers=config.num_preprocess_workers)
        
        # Initialize storage for each model
        all_features = [[] for _ in models]
        all_predictions = [[] for _ in models]
        all_logits = [[] for _ in models]
        targets = []
        
        with torch.no_grad():
            for features, label, domain_label, _ in loader:
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                targets.extend(label.cpu().numpy())
                
                # Get predictions from each model
                for i, model in enumerate(models):
                    task_output, _, encoder_output = model(signal, meta_features)
                    all_features[i].append(encoder_output.cpu().numpy())
                    all_predictions[i].extend(torch.sigmoid(task_output).cpu().numpy())
                    all_logits[i].extend(task_output.cpu().numpy())
        
        # Convert to numpy arrays
        targets = np.array(targets)
        processed_features = [np.vstack(feat_list) for feat_list in all_features]
        processed_predictions = [np.array(pred_list) for pred_list in all_predictions]
        
        # Calculate ensemble predictions (soft voting)
        ensemble_predictions = np.mean(processed_predictions, axis=0)
        
        return processed_features, processed_predictions, targets, ensemble_predictions, all_logits
    
    def calculate_metrics(predictions, targets, logits_list, dataset_name):
        """Calculate and print metrics for predictions"""
        auroc = roc_auc_score(targets, predictions)
        auprc = average_precision_score(targets, predictions)
        accuracy = accuracy_score(targets, np.round(predictions))
        f1 = f1_score(targets, np.round(predictions))
        
        # Calculate average logits for single model only
        avg_positive_logit = 0.0
        avg_negative_logit = 0.0
        if len(models) == 1:
            logits = logits_list[0]
            positive_logits = [logit for i, logit in enumerate(logits) if targets[i] == 1]
            negative_logits = [logit for i, logit in enumerate(logits) if targets[i] == 0]
            avg_positive_logit = np.mean(positive_logits) if positive_logits else 0.0
            avg_negative_logit = np.mean(negative_logits) if negative_logits else 0.0
        
        if verbose:
            print(f"\n{dataset_name} Dataset Metrics:")
            print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if len(models) == 1:
                print(f"Positive Logit: {avg_positive_logit:.4f}, Negative Logit: {avg_negative_logit:.4f}")
        
        return auroc, auprc, accuracy, f1, avg_positive_logit, avg_negative_logit
    
    def create_tsne_visualization(features, labels, title, filename, is_binary=False):
        """Create and save t-SNE visualization"""
        print(f"Performing t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_results = tsne.fit_transform(features)
        
        df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
        df_tsne['Labels'] = labels
        
        plt.figure(figsize=(12, 10))
        
        palette = ["#FDE725", "#440154"] if is_binary else sns.color_palette("hsv", len(np.unique(labels)))
        
        sns.scatterplot(x="TSNE1", y="TSNE2", hue="Labels", palette=palette, 
                       data=df_tsne, legend="full", alpha=0.7)
        
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        
        plot_path = os.path.join(config.visualisation_folder, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualization saved to {plot_path}")
    
    # Evaluate datasets
    print("Evaluating datasets...")
    
    # CODE15 dataset
    code15_features_list, code15_preds_list, code15_targets, code15_ensemble_preds, code15_logits = \
        collect_features_and_predictions(code15_dataset)
    calculate_metrics(code15_ensemble_preds, code15_targets, code15_logits, "Pretrain (CODE15)")
    
    # Combined finetune dataset
    finetune_combined_dataset = torch.utils.data.ConcatDataset([samitrop_dataset, ptbxl_dataset])
    finetune_features_list, finetune_preds_list, finetune_targets, finetune_ensemble_preds, finetune_logits = \
        collect_features_and_predictions(finetune_combined_dataset)
    calculate_metrics(finetune_ensemble_preds, finetune_targets, finetune_logits, "Finetune (SaMiTrop + PTB-XL)")
    
    # Collect external dataset features for domain visualization
    samples_per_domain = 1000
    external_features_list = [[] for _ in models]
    external_labels = []
    
    for i, ext_ds in enumerate(external_datasets):
        ext_features_list, _, ext_labels, _, _ = collect_features_and_predictions(ext_ds, samples_per_domain)
        for j, ext_features in enumerate(ext_features_list):
            external_features_list[j].append(ext_features)
        external_labels.extend([config.dann['external_datasets'][i]] * len(ext_labels))
    
    # Create visualizations for each model
    for model_idx in range(len(models)):
        model_suffix = f"_model_{model_idx + 1}" if len(models) > 1 else ""
        print(f"\nCreating visualizations for model {model_idx + 1}...")
        
        # Domain adaptation visualization
        all_domain_features = []
        all_domain_labels = []
        
        # Add CODE15 features
        num_samples_code15 = min(len(code15_features_list[model_idx]), samples_per_domain)
        indices = np.random.choice(len(code15_features_list[model_idx]), num_samples_code15, replace=False)
        all_domain_features.append(code15_features_list[model_idx][indices])
        all_domain_labels.extend(['CODE15'] * num_samples_code15)
        
        # Add SaMiTrop and PTB-XL features
        num_samitrop = len(samitrop_dataset)
        samitrop_features = finetune_features_list[model_idx][:num_samitrop]
        ptbxl_features = finetune_features_list[model_idx][num_samitrop:]
        
        num_samples_samitrop = min(len(samitrop_features), samples_per_domain)
        indices = np.random.choice(len(samitrop_features), num_samples_samitrop, replace=False)
        all_domain_features.append(samitrop_features[indices])
        all_domain_labels.extend(['SaMiTrop'] * num_samples_samitrop)
        
        num_samples_ptbxl = min(len(ptbxl_features), samples_per_domain)
        indices = np.random.choice(len(ptbxl_features), num_samples_ptbxl, replace=False)
        all_domain_features.append(ptbxl_features[indices])
        all_domain_labels.extend(['PTB-XL'] * num_samples_ptbxl)
        
        # Add external dataset features
        all_domain_features.extend(external_features_list[model_idx])
        all_domain_labels.extend(external_labels)
        
        combined_domain_features = np.vstack(all_domain_features)
        create_tsne_visualization(
            combined_domain_features, all_domain_labels,
            f't-SNE Visualization of Encoder Features by Domain (after {stage_name}){model_suffix}',
            f'dann_tsne_visualization_{stage_name}{model_suffix}.png'
        )
        
        # Chagas task visualizations
        print("Creating Chagas task visualizations...")
        
        # CODE15 Chagas visualization (balanced sampling)
        positive_indices = [i for i, label in enumerate(code15_targets) if label == 1]
        negative_indices = [i for i, label in enumerate(code15_targets) if label == 0]
        
        num_positive = len(positive_indices)
        num_negative_to_sample = min(num_positive, len(negative_indices))
        sampled_negative_indices = np.random.choice(negative_indices, num_negative_to_sample, replace=False)
        selected_indices = np.concatenate([positive_indices, sampled_negative_indices])
        np.random.shuffle(selected_indices)
        
        selected_features = code15_features_list[model_idx][selected_indices]
        selected_labels = code15_targets[selected_indices]
        
        create_tsne_visualization(
            selected_features, selected_labels,
            f't-SNE Visualization of Encoder Features for Chagas Task (Pretrain datasets, after {stage_name}){model_suffix}',
            f'pretrain_chagas_tsne_visualization_{stage_name}{model_suffix}.png',
            is_binary=True
        )
        
        # Finetune Chagas visualization (SaMiTrop vs PTB-XL)
        samitrop_targets = finetune_targets[:num_samitrop]
        ptbxl_targets = finetune_targets[num_samitrop:]
        
        positive_indices = [i for i, label in enumerate(samitrop_targets) if label == 1]
        negative_indices = [i for i, label in enumerate(ptbxl_targets) if label == 0]
        
        num_positive = len(positive_indices)
        num_negative_to_sample = min(num_positive, len(negative_indices))
        sampled_negative_indices = np.random.choice(negative_indices, num_negative_to_sample, replace=False)
        
        selected_features = np.vstack([
            samitrop_features[positive_indices],
            ptbxl_features[sampled_negative_indices]
        ])
        selected_labels = np.concatenate([
            samitrop_targets[positive_indices],
            ptbxl_targets[sampled_negative_indices]
        ])
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(selected_labels))
        selected_features = selected_features[shuffle_indices]
        selected_labels = selected_labels[shuffle_indices]
        
        create_tsne_visualization(
            selected_features, selected_labels,
            f't-SNE Visualization of Encoder Features for Chagas Task (Finetune datasets, after {stage_name}){model_suffix}',
            f'finetune_chagas_tsne_visualization_{stage_name}{model_suffix}.png',
            is_binary=True
        )
    
    print(f"\nEvaluation completed for {len(models)} model(s).")

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    models = []
    
    # Try to load ensemble models first
    for fold in range(1, 6):
        model_filename = os.path.join(model_folder, f'model_fold{fold}.pth')
        if os.path.exists(model_filename):
            try:
                checkpoint = torch.load(model_filename, map_location=config.device)
                model = HybridModel(
                    device=config.device,
                    pth_path=model_filename,
                    config=config
                )
                model.load_state_dict(checkpoint['state_dict'])
                models.append(model)
                if verbose:
                    print(f"Successfully loaded model from {model_filename}")
            except Exception as e:
                print(f"Failed to load model {model_filename}: {str(e)}")
    
    # Fallback to single model if no ensemble models found
    if not models:
        model_filename = os.path.join(model_folder, 'model.pth')
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"No model files found in {model_folder}")
        
        try:
            checkpoint = torch.load(model_filename, map_location=config.device)
            model = HybridModel(
                device=config.device,
                pth_path=model_filename,
                config=config
            )
            model.load_state_dict(checkpoint['state_dict'])
            models.append(model)
            if verbose:
                print(f"Successfully loaded model from {model_filename}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise
    
    return models[0] if len(models) == 1 else models

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Initialize filters once
    highpass_filter_params, lowpass_filter_params, notch50_filter_params, notch60_filter_params = initialize_filters()
    # Call data_preprocess to get the processed signal, meta_features, and label directly
    _, signal, meta_features, _ = data_preprocess(
        record,
        config,
        highpass_filter_params,
        lowpass_filter_params,
        notch50_filter_params,
        notch60_filter_params
    )

    # extend batch dimension
    signal = np.expand_dims(signal, axis=0)
    meta_features = np.expand_dims(meta_features, axis=0)
    
    # transfer to device
    signal = torch.from_numpy(signal).float().to(config.device)
    signal = signal.permute(0, 2, 1)  # Change to (batch_size, channels, length)
    meta_features = torch.from_numpy(meta_features).float().to(config.device)    

    # Handle single model or model list
    if isinstance(model, list):
        # Ensemble prediction
        probabilities = []
        for m in model:
            m.eval()
            with torch.no_grad():
                task_output, _, _ = m(signal, meta_features)
                prob = torch.sigmoid(task_output.view(-1)[0]).item()
                probabilities.append(prob)
        probability_output = sum(probabilities) / len(probabilities)
    else:
        # Single model prediction
        model.eval()
        with torch.no_grad():
            task_output, _, _ = model(signal, meta_features)
            probability_output = torch.sigmoid(task_output.view(-1)[0]).item()

    binary_output = probability_output > 0.5

    # delete_record_files(record)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def check_gradients(model):
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if torch.isnan(grad).any():
                print(f"WARNING: NaN gradient in {name}")
                print(f"  Gradient stats - min: {grad.min().item():.4f}, max: {grad.max().item():.4f}, mean: {grad.mean().item():.4f}")
                has_nan = True
            if torch.isinf(grad).any():
                print(f"WARNING: Inf gradient in {name}")
                print(f"  Gradient stats - min: {grad.min().item():.4f}, max: {grad.max().item():.4f}, mean: {grad.mean().item():.4f}")
                has_inf = True
    
    if has_nan or has_inf:
        print("WARNING: Model contains invalid gradients (NaN/Inf)")

def make_weights_for_balanced_classes(dataset):
    targets = [dataset[i][1] for i in range(len(dataset))]
    weights = np.zeros_like(targets, dtype=np.float32)
    weights[np.isclose(targets, 0.0)] = 1.0    # Negative class weight
    weights[np.isclose(targets, 1.0)] = config.augmentation['pos_sample_weight_multiplier']   # Positive class weight (configurable ratio)
    return weights.flatten()

# Split into CODE-15% and other records (parallel processing)
def classify_record(record_full_path):
    header = load_header(record_full_path)
    source = get_source(header)
    
    if source == 'CODE-15%':
        return (record_full_path, 'CODE15')
    elif source == 'SaMi-Trop':
        return (record_full_path, 'SaMiTrop')
    elif source == 'PTB-XL':
        return (record_full_path, 'PTBXL')
    else:
        return (record_full_path, 'Other')
