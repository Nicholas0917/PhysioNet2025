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
from models import *
from utils import *

################################################################################
#
# Global configuration. 
#
################################################################################

class Config:
    def __init__(self):
        self.model_name = 'ECGFeatureExtractor' # [ECGFeatureExtractor, ecgfounder, ResNet18, ResNet34, ResNet50]
        self.use_pretrained = True
        self.pretrain_num_epochs = 50
        self.pretrain_learning_rate = 3e-5
        self.pretrain_batch_size = 128
        self.gradient_accumulation_steps = 1 # not used in this code
        self.pretrain_early_stop_patience = 8
        self.num_epochs = 100
        self.learning_rate = 1e-6
        self.dropout_rate = 0.3
        self.net1d_dropout_rate = 0.3
        self.batch_size = 32
        self.early_stop_patience = 8
        self.num_preprocess_workers = 6
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_folder = os.getenv('CACHE_FOLDER', './tmp')
        self.pretrain_model_path = os.path.join(os.getenv('PRETRAIN_MODEL_FOLDER', '/tmp'), 'pretrain_model.pth')
        self.visualisation_folder = os.getenv('VISUALISATION_FOLDER', '/tmp')

        # DANN parameters
        self.num_domains = 8 # code15, CSPC, CSPC_extra, Chapman_Shaoxing, Georgia, Ningbo, PTB, ST_Petersburg
        self.external_datasets = ['CODE15','CSPC', 'CSPC_extra', 'Chapman_Shaoxing', 'Georgia', 'Ningbo', 'PTB', 'ST_Petersburg']
        self.dann_lambda = 0.3 # Weight for domain classification loss
        self.dann_alpha = 10.0 # Alpha for dynamic DANN lambda calculation

        # Loss parameters
        self.pretrain_focal_gamma = 2
        self.pretrain_margin = 0.1
        self.pretrain_s = 30
        self.pretrain_lmf_alpha = 0.98
        self.pretrain_lmf_beta = 0.02
        self.pretrain_label_smoothing = 0.0

        self.finetune_focal_gamma = 2
        self.finetune_margin = 0.8
        self.finetune_s = 1
        self.finetune_lmf_alpha = 0.98
        self.finetune_lmf_beta = 0.02
        self.finetune_label_smoothing = 0.2
        
        # WeightedRandomSampler parameters
        self.pos_sample_weight_multiplier = 1.0
        
        # Data augmentation parameters with probabilities
        self.use_noise_aug = True
        self.noise_std = 0.03
        self.noise_aug_prob = 0.8
        
        self.use_scaling_aug = True
        self.scaling_min = 0.5
        self.scaling_max = 2.0
        self.scaling_aug_prob = 0.8
        
        self.use_flip_aug = False
        self.flip_aug_prob = 0.2
        
        self.use_shift_aug = True
        self.shift_max_ratio = 0.8
        self.shift_aug_prob = 0.8
        
        self.use_drop_aug = True
        self.drop_max_prob = 0.02
        self.drop_aug_prob = 0.8
        
        self.add_power_noise = False
        self.power_noise_amplitude = 0.03
        self.power_noise_prob = 0.5
        
        self.use_sine_wave_aug = False
        self.sine_min_freq = 0.001
        self.sine_max_freq = 0.02
        self.sine_max_amp = 0.08
        self.sine_aug_prob = 0.3
        
        self.use_square_wave_aug = False
        self.square_min_freq = 0.001
        self.square_max_freq = 0.1
        self.square_max_amp = 0.08
        self.square_aug_prob = 0.3
        
        self.use_cutout_aug = True
        self.cutout_max_ratio = 0.2    # Max cutout ratio of signal length
        self.cutout_aug_prob = 0.5
        
        self.use_lead_mixing_aug = True
        self.lead_mixing_lambda = 0.2  # Mixing coefficient
        self.lead_mixing_prob = 0.8    # Probability of applying
        
        self.use_time_warp_aug = True
        self.time_wrap_min_hz = 450
        self.time_wrap_max_hz = 550
        self.time_wrap_prob = 0.5       # Probability of applying

        # Baseline wander augmentation parameters
        self.use_baseline_wander = True
        self.baseline_wander_min_freq = 0.05  # Hz
        self.baseline_wander_max_freq = 0.2   # Hz
        self.baseline_wander_amp_ratio = 0.2  # Amplitude ratio to signal std
        self.baseline_wander_prob = 0.3       # Probability of applying
        
        self.is_pretrain = False

    def get_meta_feature_dim(self):
        dim = 0
        if self.use_age:
            dim += 1
        if self.use_sex:
            dim += 3
        if self.use_signal_stats:
            dim += 2
        return dim

    def print_config(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>Configuration:<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Model Name: {self.model_name}")
        print(">>>>>>>>>Pretraining Parameters:<<<<<<<<<<")
        print(f"Number of Epochs: {self.pretrain_num_epochs}")
        print(f"Learning Rate: {self.pretrain_learning_rate}")
        print(f"Batch Size: {self.pretrain_batch_size}")
        print(f"Early Stop Patience: {self.pretrain_early_stop_patience}")

        print(">>>>>>>>>Training Parameters:<<<<<<<<<<")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Net1D Dropout Rate: {self.net1d_dropout_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Early Stop Patience: {self.early_stop_patience}")

        print(">>>>>>>>>Meta Features:<<<<<<<<<<")
        print(f"Use Age: {self.use_age}")
        print(f"Use Sex: {self.use_sex}")
        print(f"Use Signal Stats: {self.use_signal_stats}")
        print(f"Meta Feature Dimension: {self.get_meta_feature_dim()}")
        
        print(">>>>>>>>>DANN Parameters:<<<<<<<<<<")
        print(f"DANN Lambda: {self.dann_lambda}")
        print(f"DANN Alpha: {self.dann_alpha}")

        print(">>>>>>>>>Data Augmentation:<<<<<<<<<<")
        print(f"Use Noise Augmentation: {self.use_noise_aug}, Probability: {self.noise_aug_prob}")
        print(f"Use Scaling Augmentation: {self.use_scaling_aug}, Probability: {self.scaling_aug_prob}")
        print(f"Use Flip Augmentation: {self.use_flip_aug}, Probability: {self.flip_aug_prob}")
        print(f"Use Shift Augmentation: {self.use_shift_aug}, Max Ratio: {self.shift_max_ratio}, Probability: {self.shift_aug_prob}")
        print(f"Use Drop Augmentation: {self.use_drop_aug}, Max Probability: {self.drop_max_prob}, Probability: {self.drop_aug_prob}")
        print(f"Use 50Hz Power Noise: {self.add_power_noise}, Probability: {self.power_noise_prob}")
        print(f"Use Sine Wave Augmentation: {self.use_sine_wave_aug}, Freq Range: [{self.sine_min_freq}, {self.sine_max_freq}], Max Amp: {self.sine_max_amp}, Probability: {self.sine_aug_prob}")
        print(f"Use Square Wave Augmentation: {self.use_square_wave_aug}, Freq Range: [{self.square_min_freq}, {self.square_max_freq}], Max Amp: {self.square_max_amp}, Probability: {self.square_aug_prob}")
        print(f"Use Cutout Augmentation: {self.use_cutout_aug}, Max Ratio: {self.cutout_max_ratio}, Probability: {self.cutout_aug_prob}")
        print(f"Use Lead Mixing Augmentation: {self.use_lead_mixing_aug}, Lambda: {self.lead_mixing_lambda}, Probability: {self.lead_mixing_prob}")
        print(f"Use Time Warping Augmentation: {self.use_time_warp_aug}, Freq Range: [{self.time_wrap_min_hz}, {self.time_wrap_max_hz}], Probability: {self.time_wrap_prob}")

        print(">>>>>>>>>Loss Parameters:<<<<<<<<<<")
        print(f"Pretrain Focal Gamma: {self.pretrain_focal_gamma}")
        print(f"Pretrain Margin: {self.pretrain_margin}")
        print(f"Pretrain S: {self.pretrain_s}")
        print(f"Pretrain LMF Alpha: {self.pretrain_lmf_alpha}")
        print(f"Pretrain LMF Beta: {self.pretrain_lmf_beta}")
        print(f"Pretrain Label Smoothing: {self.pretrain_label_smoothing}")
        print(f"Finetune Focal Gamma: {self.finetune_focal_gamma}")
        print(f"Finetune Margin: {self.finetune_margin}")
        print(f"Finetune S: {self.finetune_s}")
        print(f"Finetune LMF Alpha: {self.finetune_lmf_alpha}")
        print(f"Finetune LMF Beta: {self.finetune_lmf_beta}")
        print(f"Finetune Label Smoothing: {self.finetune_label_smoothing}")
        print(f"Positive Sample Weight Multiplier: {self.pos_sample_weight_multiplier}")
        
        print(">>>>>>>>>Device:<<<<<<<<<<")
        print(f"Device: {self.device}")

config = Config()
config.print_config()
print(config.cache_folder)

# Configure CUDA/cuDNN for stability
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True

# os.environ['CUDNN_V8_API_ENABLED'] = '0'

# # if '/mnt/scratch/wmqn2362/PhysioNet25/tmp' exist
# if os.path.exists('/mnt/scratch/wmqn2362/PhysioNet25/tmp'):
#     config.cache_folder = '/mnt/scratch/wmqn2362/PhysioNet25/tmp'
# else:
#     config.cache_folder = './tmp'


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    """Train the model using the three-stage process"""
    # torch.autograd.set_detect_anomaly(True)
    # print_memory_usage("Initial Memory State in train_model")
    
    ############################################################################
    # Stage 0: Data Loading and Preprocessing
    ############################################################################
    records_relative = find_records(data_folder)
    records_full_path = [os.path.join(data_folder, r) for r in records_relative]
    # print_memory_usage("After finding records and getting full paths")
    
    # Define HDF5 file paths
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

    # Function to preprocess and write to HDF5
    def preprocess_and_write_to_hdf5(records_list, hdf5_path, dataset_name):
        if not os.path.exists(hdf5_path):
            start_time = time.time()
            print(f"Starting data preprocessing for {dataset_name} (creating HDF5)")

            if not records_list:
                print(f"No records for {dataset_name}. Skipping HDF5 creation.")
                return

            n_records = len(records_list)
            n_channels = 12
            n_samples = 5000
            meta_dim = config.get_meta_feature_dim()
            label_dim = 1
            chunk_n = os.cpu_count()

            with h5py.File(hdf5_path, 'w') as hdf5_file:
                with ProcessPoolExecutor(max_workers=cpu_num) as executor:
                    signal_chunks = (chunk_n, n_samples, n_channels)
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
                                del signal, meta_features, label

                    for fut in as_completed(pending):
                        idx, signal, meta_features, label = fut.result()
                        hdf5_file['signals'][idx] = signal
                        hdf5_file['meta_features'][idx] = meta_features
                        hdf5_file['labels'][idx] = label
                        del signal, meta_features, label

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Created new {dataset_name} data file at {hdf5_path}")
            print(f"Data preprocessing and HDF5 creation for {dataset_name} took {elapsed_time:.2f} seconds.")
        else:
            print(f"Using existing {dataset_name} data file at {hdf5_path}")

    # Preprocess and write CODE-15% data
    if not os.path.exists(code15_hdf5_path):
        preprocess_and_write_to_hdf5(code15_records_full_path, code15_hdf5_path, 'CODE15_data')
    else:
        print(f"Skipping CODE15_data preprocessing as {code15_hdf5_path} already exists.")

    # Preprocess and write SaMiTrop data
    if not os.path.exists(samitrop_hdf5_path):
        preprocess_and_write_to_hdf5(samitrop_records_full_path, samitrop_hdf5_path, 'SaMiTrop_data')
    else:
        print(f"Skipping SaMiTrop_data preprocessing as {samitrop_hdf5_path} already exists.")

    # Preprocess and write PTB-XL data
    if not os.path.exists(ptbxl_hdf5_path):
        preprocess_and_write_to_hdf5(ptbxl_records_full_path, ptbxl_hdf5_path, 'PTBXL_data')
    else:
        print(f"Skipping PTBXL_data preprocessing as {ptbxl_hdf5_path} already exists.")

    del code15_records_full_path, samitrop_records_full_path, ptbxl_records_full_path
    
    # Create datasets using the new HDF5 files
    pretrain_dataset = ECGDataset(code15_hdf5_path, is_training=True, config=config)
    
    # Create finetune datasets from SaMiTrop and PTB-XL
    samitrop_dataset = ECGDataset(samitrop_hdf5_path, is_training=True, config=config)
    ptbxl_dataset = ECGDataset(ptbxl_hdf5_path, is_training=True, config=config)
    finetune_dataset = torch.utils.data.ConcatDataset([samitrop_dataset, ptbxl_dataset])
    
    # Initialize external datasets (if needed for pretraining)
    external_datasets = []
    for dataset_name in config.external_datasets:
        external_dataset = ExternalDataset(dataset_name=dataset_name, data_folder=config.cache_folder, is_training=True, config=config)
        external_datasets.append(external_dataset)

    ############################################################################
    # Stage 1: Pretrain Model
    ############################################################################
    # Initialize model and training components
    model = HybridModel(
        device=config.device,
        config=config
    )
    # print_memory_usage("After initializing HybridModel")

    # First stage: update all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Print model parameters
    print_model_parameters(model, verbose)

    # Use create_binary_lmf_loss for pretraining
    pretrain_labels = [pretrain_dataset[i][1] for i in range(len(pretrain_dataset))]
    pretrain_neg_samples = pretrain_labels.count(0)
    pretrain_pos_samples = pretrain_labels.count(1)    
    pretrain_pos_samples_weighted = pretrain_pos_samples * config.pos_sample_weight_multiplier
    criterion = create_binary_lmf_loss(
        pos_samples=pretrain_pos_samples_weighted,
        neg_samples=pretrain_neg_samples,
        device=config.device,
        alpha=config.pretrain_lmf_alpha,
        beta=config.pretrain_lmf_beta,
        focal_gamma=config.pretrain_focal_gamma,
        ldam_margin=config.pretrain_margin,
        ldam_s=config.pretrain_s,
        label_smoothing=config.pretrain_label_smoothing
    )
    
    print_memory_usage("Before pretrain_model function call")
    model = pretrain_model(
        pretrain_dataset=pretrain_dataset,
        external_datasets=external_datasets,
        model=model,
        criterion=criterion,
        num_epochs=config.pretrain_num_epochs,
        batch_size=config.pretrain_batch_size,
        early_stop_patience=config.pretrain_early_stop_patience,
        device=config.device,
        pretrain_model_pth=os.path.join(model_folder, 'pretrain_model.pth'),
        verbose=verbose
    )
    print_memory_usage("After pretrain_model function call")
    
    ############################################################################
    # Stage 2: Evaluate pretrained model
    ############################################################################
    if verbose:
        pretrain_eval_dataset = ECGDataset(code15_hdf5_path, is_training=False, config=config)
        samitrop_eval_dataset = ECGDataset(samitrop_hdf5_path, is_training=False, config=config)
        ptbxl_eval_dataset = ECGDataset(ptbxl_hdf5_path, is_training=False, config=config)
        finetune_eval_dataset = torch.utils.data.ConcatDataset([samitrop_eval_dataset, ptbxl_eval_dataset])
        evaluate_model(model, pretrain_eval_dataset, samitrop_eval_dataset, ptbxl_eval_dataset, external_datasets, verbose, 'pretrain')
    
    ############################################################################
    # Stage 3: Finetune on target datasets
    ############################################################################
    
    # Print model parameters after freezing layers for finetuning
    print_model_parameters(model, verbose)

    # Initialize training components with BinaryLMFLoss
    # For finetuning, we also use BinaryLMFLoss
    finetune_labels = [finetune_dataset[i][1] for i in range(len(finetune_dataset))]
    finetune_neg_samples = finetune_labels.count(0)
    finetune_pos_samples = finetune_labels.count(1)

    finetune_pos_samples_weighted = finetune_pos_samples * config.pos_sample_weight_multiplier
    criterion = create_binary_lmf_loss(
        pos_samples=finetune_pos_samples_weighted,
        neg_samples=finetune_neg_samples,
        device=config.device,
        alpha=config.finetune_lmf_alpha,
        beta=config.finetune_lmf_beta,
        focal_gamma=config.finetune_focal_gamma,
        ldam_margin=config.finetune_margin,
        ldam_s=config.finetune_s,
        label_smoothing=config.finetune_label_smoothing
    )
    
    # Create optimizers and schedulers for each fold
    optimizers = []
    schedulers = []
    for _ in range(5):
        base_model_params = []
        classifier_params = []
        meta_net_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                elif 'meta_net' in name:
                    meta_net_params.append(param)
                else:
                    base_model_params.append(param)
        
        param_groups = []
        
        if base_model_params:
            param_groups.append({
                'params': base_model_params,
                'lr': config.learning_rate,
                'name': 'base_model'
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': config.learning_rate * 10,
                'name': 'classifier'
            })
        
        if meta_net_params:
            param_groups.append({
                'params': meta_net_params,
                'lr': config.learning_rate * 10,
                'name': 'meta_net'
            })
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs
        )
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    # print_memory_usage("After initializing finetune criterion, optimizers, and schedulers")
    
    kf = StratifiedKFold(n_splits=5)
    
    print_memory_usage("Before finetune_model function call")
    finetune_model(
        model=model,
        finetune_dataset=finetune_dataset,
        model_folder=model_folder,
        verbose=verbose,
        criterion=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        kf=kf
    )
    print_memory_usage("After finetune_model function call")
    
    ############################################################################
    # Stage 4: Evaluate finetuned model
    ############################################################################
    if verbose:
        pretrain_eval_dataset = ECGDataset(code15_hdf5_path, is_training=False, config=config)
        samitrop_eval_dataset = ECGDataset(samitrop_hdf5_path, is_training=False, config=config)
        ptbxl_eval_dataset = ECGDataset(ptbxl_hdf5_path, is_training=False, config=config)
        finetune_eval_dataset = torch.utils.data.ConcatDataset([samitrop_eval_dataset, ptbxl_eval_dataset])
        evaluate_model(model, pretrain_eval_dataset, samitrop_eval_dataset, ptbxl_eval_dataset, external_datasets, verbose, 'finetune')

    if verbose:
        print('Done.')
        print()

def pretrain_model(pretrain_dataset, external_datasets, model, criterion,
                  num_epochs, batch_size, early_stop_patience, device,
                  pretrain_model_pth, verbose):
    """Core pretraining logic with training loop and model saving, incorporating DANN"""
    # torch.autograd.set_detect_anomaly(True)
    os.makedirs(os.path.dirname(pretrain_model_pth), exist_ok=True)
    
    if os.path.exists(config.pretrain_model_path):
        if verbose:
            print(f"Found existing pretrained model at {config.pretrain_model_path}. Loading model and skipping pretraining.")
        model.load_state_dict(torch.load(config.pretrain_model_path, map_location=device))
        # Add the following lines to save the model after loading
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
        ext_train_size = int(0.8 * len(ext_ds))
        ext_val_size = len(ext_ds) - ext_train_size
        ext_train_dataset, ext_val_dataset = torch.utils.data.random_split(ext_ds, [ext_train_size, ext_val_size])
        
        ext_train_loader = DataLoader(ext_train_dataset,
                                      batch_size=batch_size // config.num_domains,
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
        lr=config.pretrain_learning_rate,
        weight_decay=2e-4
    )
    
    # Optimizer for domain classifier (model.domain_classifier)
    optimizer_domain_classifier = torch.optim.AdamW(
        model.domain_classifier.parameters(),
        lr=config.pretrain_learning_rate * 0.1, # Can be different from task optimizer LR
        weight_decay=2e-4
    )

    # Optimizer for encoder (model.encoder) for confusion
    optimizer_encoder_confusion = torch.optim.AdamW(
        model.encoder.parameters(), # Only encoder parameters
        lr=config.pretrain_learning_rate,
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

    # Define domain labels mapping based on config.external_datasets
    domain_labels = {name: i for i, name in enumerate(config.external_datasets)}

    best_auprc = 0.0 # This will track AUPRC for Chagas classification on CODE-15% validation set
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    def calculate_lambda(epoch, num_epochs, high=1.0, low=0.0, alpha=config.dann_alpha):
        progress = epoch / num_epochs
        return high - (high - low) * (2.0 / (1.0 + math.exp(-alpha * progress)) - 1.0)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_chagas_loss = 0.0
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
                code15_features_chagas, code15_label_chagas = next(code15_chagas_iter)
            except StopIteration:
                code15_chagas_iter = iter(code15_chagas_loader)
                code15_features_chagas, code15_label_chagas = next(code15_chagas_iter)
            
            signal_chagas, meta_chagas = code15_features_chagas
            signal_chagas = signal_chagas.to(device)
            meta_chagas = meta_chagas.to(device)
            code15_label_chagas = code15_label_chagas.to(device).view(-1, 1)

            task_output_chagas, _, _ = model(signal_chagas, meta_chagas) 
            chagas_loss = criterion(task_output_chagas, code15_label_chagas)
            
            chagas_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_task.step()
            total_chagas_loss += chagas_loss.item()
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
                    ext_features, ext_label = next(ext_iter)
                except StopIteration:
                    external_iters[j] = iter(external_train_loaders[j]) # Use external_train_loaders here
                    ext_features, ext_label = next(external_iters[j])
                
                signal_ext, meta_ext = ext_features
                all_signals_domain.append(signal_ext)
                all_metas_domain.append(meta_ext)
                all_domain_labels_combined.append(ext_label.to(device))
            
            combined_domain_signal = torch.cat(all_signals_domain, 0).to(device)
            combined_domain_meta = torch.cat(all_metas_domain, 0).to(device)
            combined_domain_target = torch.cat(all_domain_labels_combined, 0).to(device)

            # Use detached features to prevent gradient flow back to encoder
            _, domain_output_combined, _ = model(combined_domain_signal, combined_domain_meta)
            domain_loss = domain_criterion(domain_output_combined, combined_domain_target)
            
            domain_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_domain_classifier.step()
            total_domain_loss += domain_loss.item()
            train_domain_targets.extend(combined_domain_target.cpu().numpy())
            train_domain_outputs.extend(torch.argmax(domain_output_combined, dim=1).detach().cpu().numpy())

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
            dann_lambda = calculate_lambda(epoch, num_epochs, config.dann_lambda)
            confusion_loss = dann_lambda * confusion_criterion(domain_output_confusion, combined_domain_target)
            
            confusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_encoder_confusion.step()
            total_confusion_loss += confusion_loss.item()
        
        total_chagas_loss /= len(code15_chagas_loader) # Average over code15 batches
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
            for features, label in code15_val_loader:
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

        print(f'Epoch {epoch + 1}/{num_epochs}, Chagas Loss: {total_chagas_loss:.4f}, Domain Loss: {total_domain_loss:.4f}, Confusion Loss: {total_confusion_loss:.4f}, Valid Loss (Chagas): {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
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
                for features, label in ext_val_loader: # label is not used for external datasets in this context
                    signal, meta_features = features
                    signal = signal.to(device)
                    meta_features = meta_features.to(device)
                    
                    _, domain_output, _ = model(signal, meta_features)
                    
                    domain_label_ext_val = label.to(device) # Directly use label from loader
                    ext_val_domain_targets.extend(domain_label_ext_val.cpu().numpy())
                    ext_val_domain_outputs.extend(torch.argmax(domain_output, dim=1).detach().cpu().numpy())
            
            ext_domain_accuracy = accuracy_score(ext_val_domain_targets, ext_val_domain_outputs)
            print(f'Valid Domain Accuracy ({config.external_datasets[j]}): {ext_domain_accuracy:.4f}')
        print('\n')

        scheduler_task.step()
        scheduler_domain_classifier.step()
        scheduler_encoder_confusion.step()

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
        
        train_subset = Subset(finetune_dataset, train_idx)
        val_subset = Subset(finetune_dataset, val_idx)

        # Create weighted sampler for imbalanced data
        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(train_subset, 
                                batch_size=config.batch_size, 
                                sampler=train_sampler,
                                num_workers=config.num_preprocess_workers,
                                drop_last=True)
        val_loader = DataLoader(val_subset, 
                              batch_size=config.batch_size, 
                              shuffle=False,
                              num_workers=config.num_preprocess_workers,
                              drop_last=True)

        best_loss = float('inf')
        best_auprc = 0.0
        best_epoch = 0
        start_time = time.time()

        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            train_targets = []
            train_outputs = []
            if verbose:
                pos_logits_sum = 0.0
                neg_logits_sum = 0.0
                pos_count = 0
                neg_count = 0
            
            for i, (features, label) in enumerate(train_loader):
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                label = label.to(config.device)

                optimizers[fold].zero_grad()
                # with autocast:  # Disabled mixed precision training
                
                task_output, _, _ = model(signal, meta_features)
                
                # Reshape label to match output shape [batch_size, 1]
                label_reshaped = label.view(-1, 1)
                loss = criterion(task_output, label_reshaped)
                
                # Gradient clipping
                # scaler.scale(loss).backward()
                loss.backward()
                check_gradients(model)
                # scaler.unscale_(optimizers[fold])
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizers[fold])
                optimizers[fold].step()
                # scaler.update()
                
                train_loss += loss.item()
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

            schedulers[fold].step()

            # Calculate training metrics
            train_auroc = roc_auc_score(train_targets, train_outputs)
            train_auprc = average_precision_score(train_targets, train_outputs)
            train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
            train_f1 = f1_score(train_targets, np.round(train_outputs))

            # Validation
            model.eval()
            val_loss = 0.0
            val_targets = []
            val_outputs = []
            if verbose:
                val_pos_logits_sum = 0.0
                val_neg_logits_sum = 0.0
                val_pos_count = 0
                val_neg_count = 0

            with torch.no_grad():
                for i, (features, label) in enumerate(val_loader):
                    signal, meta_features = features
                    signal = signal.to(config.device)
                    meta_features = meta_features.to(config.device)
                    label = label.to(config.device)

                    # with autocast:
                    task_output, _, _ = model(signal, meta_features)
                    # Reshape label to match output shape [batch_size, 1]
                    label_reshaped = label.view(-1, 1)
                    loss = criterion(task_output, label_reshaped)
                    
                    val_loss += loss.item()
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
            
            val_auroc = roc_auc_score(val_targets, val_outputs)
            val_auprc = average_precision_score(val_targets, val_outputs)
            val_accuracy = accuracy_score(val_targets, np.round(val_outputs))
            val_f1 = f1_score(val_targets, np.round(val_outputs))

            epoch_duration = time.time() - epoch_start_time
            
            if verbose:
                print(f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
                print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
                print(f'Valid AUROC: {val_auroc:.4f}, Valid AUPRC: {val_auprc:.4f}, Valid Accuracy: {val_accuracy:.4f}, Valid F1: {val_f1:.4f}\n')
                # Calculate epoch averages
                epoch_pos_logit = pos_logits_sum / pos_count if pos_count > 0 else 0.0
                epoch_neg_logit = neg_logits_sum / neg_count if neg_count > 0 else 0.0
                
                print(f"Epoch {epoch + 1} Averages (Train) - Positive Logit: {epoch_pos_logit:.4f}, Negative Logit: {epoch_neg_logit:.4f}")
                # Calculate validation epoch averages
                val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
                val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
                
                print(f"Valid Averages (Valid) - Positive Logit: {val_epoch_pos_logit:.4f}, Negative Logit: {val_epoch_neg_logit:.4f}")

            # Early stopping based on AUPRC
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.early_stop_patience:
                    if verbose:
                        print(f"Early stopping: Valid AUPRC not improved for {config.early_stop_patience} epochs")
                    break

            # del train_targets, train_outputs, val_targets, val_outputs
            # torch.cuda.empty_cache()
            # gc.collect()
        
        end_time = time.time()
        if verbose:
            print(f'Fold {fold + 1} finished. Best Valid AUPRC: {best_auprc:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

        # Save finetuned model
        os.makedirs(model_folder, exist_ok=True)
        save_model(model_folder, best_model, config, fold=fold+1)
        if verbose:
            print(f'Finetuning completed for fold {fold+1}')

def evaluate_model(model, code15_dataset, samitrop_dataset, ptbxl_dataset, external_datasets, verbose, stage_name):
    """Evaluate model performance on both pretrain and finetune datasets"""
    model.eval()
    # mkdir visualisation_folder
    os.makedirs(config.visualisation_folder, exist_ok=True)

    def collect_encoder_features_and_labels(dataset, num_samples_to_take=None):
        features_list = []
        labels_list = []
        if dataset is None or len(dataset) == 0:
            return np.array([]), np.array([])

        if num_samples_to_take is not None:
            indices = np.random.choice(len(dataset), min(len(dataset), num_samples_to_take), replace=False)
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_preprocess_workers)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_preprocess_workers)

        with torch.no_grad():
            for features, label in loader:
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)

                encoder_output = model.encoder(signal, meta_features)
                features_list.append(encoder_output.cpu().numpy())
                labels_list.extend(label.cpu().numpy())

        if not features_list:
            return np.array([]), np.array([])
        return np.vstack(features_list), np.array(labels_list)

    def evaluate_dataset(dataset, dataset_name):
        loader = DataLoader(dataset,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_preprocess_workers)
        
        # Evaluation
        outputs = []
        targets = []
        all_logits = []
        with torch.no_grad():
            for features, label in loader:
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                label = label.to(config.device)

                task_output, _, _ = model(signal, meta_features)
                outputs.extend(torch.sigmoid(task_output).cpu().numpy())
                targets.extend(label.cpu().numpy())
                all_logits.extend(task_output.cpu().numpy())
        
        # Calculate metrics
        auroc = roc_auc_score(targets, outputs)
        auprc = average_precision_score(targets, outputs)
        accuracy = accuracy_score(targets, np.round(outputs))
        f1 = f1_score(targets, np.round(outputs))
        
        if verbose:
            # Calculate positive and negative logits
            positive_logits = [logit for i, logit in enumerate(all_logits) if targets[i] == 1]
            negative_logits = [logit for i, logit in enumerate(all_logits) if targets[i] == 0]

            avg_positive_logit = np.mean(positive_logits) if positive_logits else 0.0
            avg_negative_logit = np.mean(negative_logits) if negative_logits else 0.0

            print(f"\n{dataset_name} Dataset Metrics:")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Positive Logit: {avg_positive_logit:.4f}")
            print(f"Negative Logit: {avg_negative_logit:.4f}")

        # del outputs, targets, all_logits
        # torch.cuda.empty_cache()
        # gc.collect()
    
    # Evaluate on code15 dataset (formerly pretrain_dataset)
    evaluate_dataset(code15_dataset, "Pretrain")
    
    # Concatenate samitrop_dataset and ptbxl_dataset for finetune evaluation
    finetune_combined_dataset = torch.utils.data.ConcatDataset([samitrop_dataset, ptbxl_dataset])
    evaluate_dataset(finetune_combined_dataset, "Finetune")

    if verbose:
        print("\nStarting DANN visualization...")
        
        # Collect features and domain labels for t-SNE visualization
        all_features = []
        all_domain_labels = []
        samples_per_domain = 1000 # User specified 1000 samples per domain

        # Prepare a list of all datasets to sample from, with their corresponding domain names
        datasets_for_tsne = []

        # Add SaMiTrop dataset
        if samitrop_dataset is not None and len(samitrop_dataset) > 0:
            datasets_for_tsne.append({'name': 'SaMiTrop', 'dataset': samitrop_dataset})

        # Add PTB-XL dataset
        if ptbxl_dataset is not None and len(ptbxl_dataset) > 0:
            datasets_for_tsne.append({'name': 'PTB-XL', 'dataset': ptbxl_dataset})

        # Add external datasets
        for i, ext_ds in enumerate(external_datasets):
            if ext_ds is not None and len(ext_ds) > 0:
                datasets_for_tsne.append({'name': config.external_datasets[i], 'dataset': ext_ds})

        if not datasets_for_tsne:
            print("No datasets available for DANN visualization. Skipping t-SNE.")
            return

        # Collect features and domain labels
        for i, item in enumerate(datasets_for_tsne):
            ds_name = item['name']
            dataset = item['dataset']
            num_samples_to_take = min(len(dataset), samples_per_domain)
            
            if num_samples_to_take > 0:
                # Use the unified function to collect features and labels
                features_from_dataset, labels_from_dataset = collect_encoder_features_and_labels(dataset, num_samples_to_take)
                all_features.append(features_from_dataset)
                # Convert numerical labels to domain names
                domain_names_from_dataset = [ds_name] * len(labels_from_dataset) # Assign domain name directly
                all_domain_labels.extend(domain_names_from_dataset)

        if not all_features:
            print("No features collected for DANN visualization. Skipping t-SNE.")
            return

        all_features = np.vstack(all_features)

        # Perform t-SNE
        print(f"Performing t-SNE on {len(all_features)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_results = tsne.fit_transform(all_features)

        # Create DataFrame for plotting
        df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
        df_tsne['Domain'] = all_domain_labels

        # Plotting
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x="TSNE1", y="TSNE2",
            hue="Domain",
            palette=sns.color_palette("hsv", len(np.unique(all_domain_labels))),
            data=df_tsne,
            legend="full",
            alpha=0.7
        )
        plt.title(f't-SNE Visualization of Encoder Features by Domain ({stage_name})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)

        # Save the plot
        output_folder = config.visualisation_folder
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, f'dann_tsne_visualization_{stage_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"t-SNE visualization saved to {plot_path}")

        # Chagas visualization for Pretrain dataset
        print("\nStarting Chagas task feature visualization for CODE15 dataset...")
        code15_features, code15_labels = collect_encoder_features_and_labels(code15_dataset, num_samples_to_take=None)
        
        if code15_features.size > 0:
            print(f"Performing t-SNE on {len(code15_features)} samples for CODE15 Chagas task visualization...")
            tsne_results_code15_chagas = tsne.fit_transform(code15_features)

            df_tsne_code15_chagas = pd.DataFrame(tsne_results_code15_chagas, columns=['TSNE1', 'TSNE2'])
            df_tsne_code15_chagas['Chagas Label'] = code15_labels

            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x="TSNE1", y="TSNE2",
                hue="Chagas Label",
                palette="coolwarm",
                data=df_tsne_code15_chagas,
                legend="full",
                alpha=0.7
            )
            plt.title(f't-SNE Visualization of Encoder Features for Pretrain Chagas Task ({stage_name})')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True)

            plot_path_code15_chagas = os.path.join(config.visualisation_folder, f'code15_chagas_tsne_visualization_{stage_name}.png')
            plt.savefig(plot_path_code15_chagas)
            plt.close()
            print(f"Pretrain Chagas task t-SNE visualization saved to {plot_path_code15_chagas}")
        else:
            print("No features collected for Pretrain Chagas task visualization. Skipping t-SNE.")

        # Chagas visualization for Finetune dataset
        print("\nStarting Chagas task feature visualization for Finetune dataset...")
        finetune_combined_features, finetune_combined_labels = collect_encoder_features_and_labels(finetune_combined_dataset)

        if finetune_combined_features.size > 0:
            print(f"Performing t-SNE on {len(finetune_combined_features)} samples for Finetune Chagas task visualization...")
            tsne_results_finetune_combined_chagas = tsne.fit_transform(finetune_combined_features)

            df_tsne_finetune_combined_chagas = pd.DataFrame(tsne_results_finetune_combined_chagas, columns=['TSNE1', 'TSNE2'])
            df_tsne_finetune_combined_chagas['Chagas Label'] = finetune_combined_labels

            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x="TSNE1", y="TSNE2",
                hue="Chagas Label",
                palette="coolwarm",
                data=df_tsne_finetune_combined_chagas,
                legend="full",
                alpha=0.7
            )
            plt.title(f't-SNE Visualization of Encoder Features for Finetune Chagas Task ({stage_name})')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True)

            plot_path_finetune_combined_chagas = os.path.join(config.visualisation_folder, f'finetune_combined_chagas_tsne_visualization_{stage_name}.png')
            plt.savefig(plot_path_finetune_combined_chagas)
            plt.close()
            print(f"Finetune Chagas task t-SNE visualization saved to {plot_path_finetune_combined_chagas}")
        else:
            print("No features collected for Finetune Chagas task visualization. Skipping t-SNE.")

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
    weights[np.isclose(targets, 1.0)] = config.pos_sample_weight_multiplier   # Positive class weight (configurable ratio)
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
