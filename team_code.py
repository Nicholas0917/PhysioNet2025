# todo:
# Add ECG Augmentation

#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import math
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import psutil
from net1d import Net1D, Swish

from helper_code import *

################################################################################
#
# Global configuration. 
#
################################################################################

class Config:
    def __init__(self):
        self.model_name = 'ecgfounder'
        self.use_pretrained = True
        self.pretrain_num_epochs = 50
        self.pretrain_learning_rate = 1e-4  # Increased learning rate
        self.pretrain_batch_size = 256  # Increased with gradient accumulation
        self.gradient_accumulation_steps = 4  # For effective batch size of 256
        self.pretrain_early_stop_patience = 5
        self.num_epochs = 100
        self.learning_rate = 1e-5  # Increased learning rate
        self.dropout_rate = 0.3
        self.net1d_dropout_rate = 0.3
        self.batch_size = 32
        self.early_stop_patience = 8
        self.num_preprocess_workers = 2  # Keep at 1 to limit memory usage
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_folder = '/mnt/scratch/wmqn2362/PhysioNet25/tmp'

        # Focal Loss parameters
        self.pretrain_focal_alpha = 0.8
        self.pretrain_focal_gamma = 3
        self.finetune_focal_alpha = 0.8
        self.finetune_focal_gamma = 2
        
        # Data augmentation parameters with probabilities
        self.use_noise_aug = True
        self.noise_std = 0.03
        self.noise_aug_prob = 0.5
        
        self.use_scaling_aug = True
        self.scaling_min = 0.5
        self.scaling_max = 2.0
        self.scaling_aug_prob = 0.5
        
        self.use_flip_aug = False
        self.flip_aug_prob = 0.2
        
        self.use_shift_aug = True
        self.shift_max_ratio = 0.8
        self.shift_aug_prob = 0.3
        
        self.use_drop_aug = False
        self.drop_max_prob = 0.02
        self.drop_aug_prob = 0.3
        
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
        self.cutout_aug_prob = 0.3
        
        self.use_lead_mixing_aug = True
        self.lead_mixing_lambda = 0.2  # Mixing coefficient
        self.lead_mixing_prob = 0.3    # Probability of applying
        
        self.use_time_warp_aug = False
        self.time_wrap_min_hz = 450
        self.time_wrap_max_hz = 550
        self.time_wrap_prob = 0.3       # Probability of applying

        # Baseline wander augmentation parameters
        self.use_baseline_wander = True
        self.baseline_wander_min_freq = 0.05  # Hz
        self.baseline_wander_max_freq = 0.2   # Hz
        self.baseline_wander_amp_ratio = 0.2  # Amplitude ratio to signal std
        self.baseline_wander_prob = 0.3       # Probability of applying

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

        print(">>>>>>>>>Device:<<<<<<<<<<")
        print(f"Device: {self.device}")

config = Config()
config.print_config()

# Set deterministic CUDA backend
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# if '/mnt/scratch/wmqn2362/PhysioNet25/tmp' exist
if os.path.exists('/mnt/scratch/wmqn2362/PhysioNet25/tmp'):
    config.cache_folder = '/mnt/scratch/wmqn2362/PhysioNet25/tmp'
else:
    config.cache_folder = './tmp'

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Enable anomaly detection for debugging NaN/Inf values
    torch.autograd.set_detect_anomaly(True)

    ############################################################################
    # Load the data.
    start_time = time.time()
    records = find_records(data_folder)
    end_time = time.time()
    print(f"Load the data time: {end_time - start_time:.4f}seconds")
    num_records = len(records)
    
    print(f'Total number of records: {num_records}')
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # divide the records according to the source using parallel processing
    code15_records = []
    PTBXL_records = []
    SaMiTrop_records = []
    
    def process_record(record):
        record_path = os.path.join(data_folder, record)
        header = load_header(record_path)
        source = get_source(header)
        return (record_path, source)
    
    with ThreadPoolExecutor(max_workers=config.num_preprocess_workers) as executor:
        results = list(executor.map(process_record, records))
    
    for record_path, source in results:
        if source == 'PTB-XL':
            PTBXL_records.append(record_path)
        elif source == 'CODE-15%':
            code15_records.append(record_path)
        elif source == 'SaMi-Trop':
            SaMiTrop_records.append(record_path)
        else:
            raise ValueError('Invalid source.')
      
    Code15_records_pretrain = code15_records
    Code15_records_finetune = []

    # Pretrain
    if verbose:
        print('Pretraining the model on the CODE%15 data...')

    # print("Pretrain Datastes Size: ",len(Code15_records_pretrain))
    # print_memory_usage("Before data preprocessing")
    start_time = time.time()  # Record start time
    
    # Parallel data preprocessing
    # with ThreadPoolExecutor(max_workers=4) as executor:
    with ThreadPoolExecutor(max_workers=config.num_preprocess_workers) as executor:
        list(executor.map(data_preprocess, Code15_records_pretrain))
    
    end_time = time.time()    # Record end time
    # print_memory_usage("After data preprocessing")
    # print(f"Data preprocessing time: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    dataset = ECGDataset(Code15_records_pretrain, is_training=True)
    end_time = time.time()
    # print(f"ECGDataset initialization time: {end_time - start_time:.4f} seconds")
    # print_memory_usage("After creating dataset")
    ############################################################################
    # Pretrain the models.

    num_epochs = config.pretrain_num_epochs
    learning_rate = config.pretrain_learning_rate
    batch_size = config.pretrain_batch_size
    early_stop_patience = config.pretrain_early_stop_patience
    device = config.device

    # Initialize model with pretrained weights
    model = HybridModel(
        device=device,
        pth_path='/users/wmqn2362/PhysioNet2025/Founder_PhysioNet/12_lead_ECGFounder.pth'
    )
    
    # First stage: update all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Initialize FocalLoss with config parameters for pretraining
    criterion = FocalLoss(
        alpha=config.pretrain_focal_alpha,
        gamma=config.pretrain_focal_gamma,
        logits=True
    )
    # criterion = SampleWeightedLoss(beta=1.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=2e-4
    )
    # Warmup implementation using LambdaLR
    warmup_epochs = int(num_epochs * 0.1)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + (epoch / warmup_epochs) * 0.9  # Linear warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))  # After warmup
    
    # Use LambdaLR for warmup phase
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Disabled ReduceLROnPlateau per request, using only Warmup + Cosine Annealing
    # reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',
    #     patience=3,
    #     factor=0.5,
    #     threshold=0.001
    # )
    scaler = torch.amp.GradScaler('cuda')
    
    # Enable automatic mixed precision
    autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    kf = StratifiedKFold(n_splits=3)

    def make_weights_for_balanced_classes(dataset):
        targets = [dataset[i][1] for i in range(len(dataset))]
        weights = np.zeros_like(targets, dtype=np.float32)
        weights[np.isclose(targets, 0.0)] = 1.0
        weights[np.isclose(targets, 1.0)] = 30.0
        return weights

    X = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)           
            
        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(train_subset, 
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=config.num_preprocess_workers)
        
        val_loader = DataLoader(val_subset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=config.num_preprocess_workers)

        best_loss = float('inf')
        best_auprc = 0.0  # Initialize best AUPRC score
        best_epoch = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            train_targets = []
            train_outputs = []
            for i, (features, label) in enumerate(train_loader):
                signal, meta_features = features
                
                # Check for NaN in input data
                if torch.isnan(signal).any():
                    print("WARNING: Training signal contains NaN values")
                if torch.isnan(meta_features).any():
                    print("WARNING: Training meta_features contains NaN values")
                if torch.isnan(label).any():
                    print("WARNING: Training label contains NaN values")
                
                signal = signal.to(device)
                meta_features = meta_features.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                with autocast:
                    output = model(signal, meta_features)
                    loss = criterion(output, label)
                # Gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * config.gradient_accumulation_steps

                train_targets.extend(label.cpu().numpy())
                train_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())

            train_loss /= len(train_loader)

            train_auroc = roc_auc_score(train_targets, np.round(train_outputs))
            train_auprc = average_precision_score(train_targets, np.round(train_outputs))
            train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
            train_f1 = f1_score(train_targets, np.round(train_outputs))

            model.eval()
            val_loss = 0.0
            val_targets = []
            val_outputs = []
            with torch.no_grad():
                for i, (features, label) in enumerate(val_loader):
                    signal, meta_features = features
                    signal = signal.to(device)
                    meta_features = meta_features.to(device)
                    label = label.to(device)

                    with autocast:
                        output = model(signal, meta_features)
                        loss = criterion(output, label)
                    
                    val_loss += loss.item()
                    current_targets = label.cpu().numpy()
                    current_outputs = torch.sigmoid(output).detach().cpu().numpy()
                    val_targets.extend(current_targets)
                    val_outputs.extend(current_outputs)

            val_loss /= len(val_loader)
            
            # Verify data before metrics
            val_outputs_arr = np.array(val_outputs)
            val_targets_arr = np.array(val_targets)
            
            # Check for NaN values
            if np.isnan(val_outputs_arr).any():
                print(f"WARNING: val_outputs contains {np.isnan(val_outputs_arr).sum()} NaN values")
            if np.isnan(val_targets_arr).any():
                print(f"WARNING: val_targets contains {np.isnan(val_targets_arr).sum()} NaN values")

            try:
                if len(np.unique(val_targets_arr)) < 2:
                    print("WARNING: Only one class present in validation targets")
                    val_auroc = 0.5
                else:
                    val_auroc = roc_auc_score(val_targets_arr, val_outputs_arr)
                
                val_auprc = average_precision_score(val_targets_arr, val_outputs_arr)
                val_accuracy = accuracy_score(val_targets_arr, np.round(val_outputs_arr))
                val_f1 = f1_score(val_targets_arr, np.round(val_outputs_arr))
            except ValueError as e:
                print(f"Error calculating metrics: {str(e)}")
                val_auroc = 0.5
                val_auprc = 0.5
                val_accuracy = 0.5
                val_f1 = 0.0
            
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            # Disabled ReduceLROnPlateau step per request
            # else:
            #     reduce_lr_scheduler.step(val_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
            print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
            print(f'Valid AUROC: {val_auroc:.4f}, Valid AUPRC: {val_auprc:.4f}, Valid Accuracy: {val_accuracy:.4f}, Valid F1: {val_f1:.4f}\n')

            # Track best AUPRC score for early stopping
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_model = model.state_dict()  # Save state_dict only
                # Reset counter since we got a better AUPRC
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping: Valid AUPRC not improved for {early_stop_patience} epochs")
                    break

        end_time = time.time()
        print(f'Fold {fold + 1} finished. Best Valid AUPRC: {best_auprc:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

    # for record in Code15_records_pretrain:
    #     delete_record_files(record)

    # print("\nAfter pretraining:")
    # print_memory_usage()
    
    ############################################################################

    # Evaluate on pretrain dataset
    print("\nEvaluating on pretrain dataset...")
    model.eval()
    pretrain_outputs = []
    pretrain_targets = []
    
    pretrain_loader = DataLoader(ECGDataset(Code15_records_pretrain, is_training=False),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=config.num_preprocess_workers)
    
    with torch.no_grad():
        for features, label in pretrain_loader:
            signal, meta_features = features
            signal = signal.to(device)
            meta_features = meta_features.to(device)
            label = label.to(device)
            
            output = model(signal, meta_features)
            pretrain_outputs.extend(torch.sigmoid(output).cpu().numpy())
            pretrain_targets.extend(label.cpu().numpy())
    
    # Calculate metrics
    auroc = roc_auc_score(pretrain_targets, pretrain_outputs)
    auprc = average_precision_score(pretrain_targets, pretrain_outputs)
    accuracy = accuracy_score(pretrain_targets, np.round(pretrain_outputs))
    f1 = f1_score(pretrain_targets, np.round(pretrain_outputs))
    
    # Calculate loss - convert to numpy array first for better performance
    pretrain_outputs_arr = np.array(pretrain_outputs, dtype=np.float32)
    pretrain_targets_arr = np.array(pretrain_targets, dtype=np.float32)
    criterion = FocalLoss(
        alpha=config.finetune_focal_alpha,
        gamma=config.finetune_focal_gamma, 
        logits=True
    )
    all_loss = criterion(torch.from_numpy(pretrain_outputs_arr), 
                        torch.from_numpy(pretrain_targets_arr)).item()
    
    print(f"Pretrain Dataset Metrics:")
    print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"Loss: {all_loss:.4f}\n")

    # Evaluate on finetune dataset
    finetune_records = PTBXL_records + SaMiTrop_records + Code15_records_finetune
    for record in finetune_records:
        data_preprocess(record)

    print("\nEvaluating on finetune dataset...")
    model.eval()
    finetune_outputs = []
    finetune_targets = []
    
    finetune_loader = DataLoader(ECGDataset(finetune_records, is_training=False),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=config.num_preprocess_workers)
    
    with torch.no_grad():
        for features, label in finetune_loader:
            signal, meta_features = features
            signal = signal.to(device)
            meta_features = meta_features.to(device)
            label = label.to(device)
            
            output = model(signal, meta_features)
            finetune_outputs.extend(torch.sigmoid(output).cpu().numpy())
            finetune_targets.extend(label.cpu().numpy())
    
    # Calculate metrics
    auroc = roc_auc_score(finetune_targets, finetune_outputs)
    auprc = average_precision_score(finetune_targets, finetune_outputs)
    accuracy = accuracy_score(finetune_targets, np.round(finetune_outputs))
    f1 = f1_score(finetune_targets, np.round(finetune_outputs))
    
    # Calculate loss - convert to numpy array first for better performance
    finetune_outputs_arr = np.array(finetune_outputs, dtype=np.float32)
    finetune_targets_arr = np.array(finetune_targets, dtype=np.float32)
    criterion = FocalLoss(
        alpha=config.finetune_focal_alpha,
        gamma=config.finetune_focal_gamma,
        logits=True
    )
    all_loss = criterion(torch.from_numpy(finetune_outputs_arr),
                        torch.from_numpy(finetune_targets_arr)).item()
    
    print(f"Finetune Dataset Metrics:")
    print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"Loss: {all_loss:.4f}\n")

    ############################################################################
    # fine-tune stage
    if verbose:
        print('Training the model on the fine-tune data...')
    
    print("Fine-tune Datastes Size: ",len(finetune_records))
    dataset = ECGDataset(finetune_records, is_training=True)
    # print("\nAfter loading pretrain dataset:")
    # print_memory_usage()
    
    # Train the models.
    # Define the parameters using config.
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    early_stop_patience = config.early_stop_patience
    device = config.device

    # Second stage: freeze all except classifier
    if len(finetune_records) > 0:
        for name, param in model.named_parameters():
            if 'classifier' in name:  # Only update classifier
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    criterion = FocalLoss(
        alpha=config.finetune_focal_alpha,
        gamma=config.finetune_focal_gamma,
        logits=True
    )
    # criterion = SampleWeightedLoss(beta=1.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')
    kf = StratifiedKFold(n_splits=5)

    X = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    # for fold, (train_idx, val_idx) in enumerate(kf.split(records)):
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=config.num_preprocess_workers)
        train_loader = DataLoader(train_subset, 
                                  batch_size=batch_size, 
                                  sampler=train_sampler,
                                  num_workers=config.num_preprocess_workers,
                                  drop_last=True  # Ensure complete batches
                                  )
        val_loader = DataLoader(val_subset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=config.num_preprocess_workers,
                                drop_last=True  # Ensure complete batches
                                )

        best_loss = float('inf')
        best_auprc = 0.0  # Initialize best AUPRC score
        best_epoch = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            train_targets = []
            train_outputs = []
            for i, (features, label) in enumerate(train_loader):
                signal, meta_features = features
                signal = signal.to(device)
                meta_features = meta_features.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                with autocast:
                    output = model(signal, meta_features)
                    loss = criterion(output, label)
                # Gradient clipping
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

                train_targets.extend(label.cpu().numpy())
                train_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())

            train_loss /= len(train_loader)
            scheduler.step()

            train_auroc = roc_auc_score(train_targets, np.round(train_outputs))
            train_auprc = average_precision_score(train_targets, np.round(train_outputs))
            train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
            train_f1 = f1_score(train_targets, np.round(train_outputs))

            model.eval()
            val_loss = 0.0
            val_targets = []
            val_outputs = []
            with torch.no_grad():
                for i, (features, label) in enumerate(val_loader):
                    signal, meta_features = features
                    signal = signal.to(device)
                    meta_features = meta_features.to(device)
                    label = label.to(device)

                    with autocast:
                        output = model(signal, meta_features)
                        loss = criterion(output, label)
                    val_loss += loss.item()

                    val_targets.extend(label.cpu().numpy())
                    val_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())

            val_loss /= len(val_loader)
            val_auroc = roc_auc_score(val_targets, np.round(val_outputs))
            val_auprc = average_precision_score(val_targets, np.round(val_outputs))
            val_accuracy = accuracy_score(val_targets, np.round(val_outputs))
            val_f1 = f1_score(val_targets, np.round(val_outputs))

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
            print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
            print(f'Valid AUROC: {val_auroc:.4f}, Valid AUPRC: {val_auprc:.4f}, Valid Accuracy: {val_accuracy:.4f}, Valid F1: {val_f1:.4f}\n')

            # Track best AUPRC score for early stopping
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_model = model.state_dict()
                # Reset counter since we got a better AUPRC
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping: Valid AUPRC not improved for {early_stop_patience} epochs")
                    break
    
        end_time = time.time()
        print(f'Fold {fold + 1} finished. Best Valid AUPRC: {best_auprc:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

    # for record in finetune_records:
    #     delete_record_files(record)
    ############################################################################
    # Save the best model for this fold
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, best_model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # Check model directory
    model_dir = os.path.join(model_folder, 'Model')
    model_filename = os.path.join(model_dir, 'model.pth')
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file {model_filename} not found")
    
    try:
        checkpoint = torch.load(model_filename, map_location=config.device)
        
        # Create and initialize model
        model = HybridModel(
            device=config.device,
            pth_path=model_filename
        )
        
        # Load fine-tuned weights
        model.load_state_dict(checkpoint['state_dict'])
        
        if verbose:
            print(f"Successfully loaded model from {model_filename}")
        return model
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    
    data_preprocess(record)

    base_name = os.path.splitext(os.path.basename(record))[0]
    signal_path = os.path.join(config.cache_folder, f"{base_name}_signal.npy")
    signal = np.load(signal_path).astype(np.float32)

    # Load meta data from original record
    header = load_header(record)
    age = get_age(header) if config.use_age else 0
    sex = get_sex(header) if config.use_sex else 'Unknown'
    
    one_hot_encoding_sex = np.zeros(3, dtype=np.bool_)
    if sex == 'Female':
        one_hot_encoding_sex[0] = True
    elif sex == 'Male':
        one_hot_encoding_sex[1] = True
    else:
        one_hot_encoding_sex[2] = True

    # get meta features
    meta_features = np.empty(config.get_meta_feature_dim(), dtype=np.float32)
    ptr = 0
    
    if config.use_age:
        meta_features[ptr] = age
        ptr += 1
    if config.use_sex:
        meta_features[ptr:ptr+3] = one_hot_encoding_sex
        ptr += 3
    if config.use_signal_stats:
        valid_samples = np.isfinite(signal).sum()
        meta_features[ptr] = np.nanmean(signal) if valid_samples > 0 else 0.0
        meta_features[ptr+1] = np.nanstd(signal) if valid_samples > 1 else 0.0
        ptr += 2

    # extend batch dimension
    signal = np.expand_dims(signal, axis=0)
    meta_features = np.expand_dims(meta_features, axis=0)
    
    # transfer to device
    signal = torch.from_numpy(signal).to(config.device)
    meta_features = torch.from_numpy(meta_features).to(config.device)    

    # Set model to evaluation mode
    model.eval()

    # Get the model outputs.
    with torch.no_grad():
        probability_output = model(signal, meta_features)
    probability_output = torch.sigmoid(probability_output.view(-1)[0]).detach().cpu().numpy().item()
    binary_output = probability_output > 0.5

    # delete_record_files(record)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def print_memory_usage(extra_info=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    print(f"\n====== Memory Usage {extra_info} ======")
    print(f"Process RSS: {mem_info.rss / 1024 ** 3:.2f} GB")
    print(f"Process VMS: {mem_info.vms / 1024 ** 3:.2f} GB")
    print(f"System Available: {vm.available / 1024 ** 3:.2f} GB / {vm.total / 1024 ** 3:.2f} GB")
    print(f"Memory Used %: {vm.percent}%")
    print("===================================\n")

# Extract your features.
def data_preprocess(record):
    os.makedirs(config.cache_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(record))[0]
    
    signal_path = os.path.join(config.cache_folder, f'{base_name}_signal.npy')
    if os.path.exists(signal_path):
        return

    header = load_header(record)
    source = get_source(header)
    age = get_age(header) if config.use_age else 0
    sex = get_sex(header) if config.use_sex else 'Unknown'
    
    one_hot_encoding_sex = np.zeros(3, dtype=np.bool_)
    if sex == 'Female':
        one_hot_encoding_sex[0] = True
    elif sex == 'Male':
        one_hot_encoding_sex[1] = True
    else:
        one_hot_encoding_sex[2] = True

    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    signal = signal.astype(np.float32)
    
    # Check for NaN/Inf in raw signal
    if np.isnan(signal).any() or np.isinf(signal).any():
        print("WARNING: Raw signal contains NaN/Inf values")

    # Standardize all data to 500Hz
    original_fs = get_sampling_frequency(header)
    target_fs = 500
    if original_fs != target_fs:
        target_length = int(signal.shape[0] * target_fs / original_fs)
        try:
            resampled_signal = resample(signal, target_length, axis=0).astype(np.float32)
            signal = resampled_signal
        except Exception as e:
            print(f"Error in resampling: {str(e)}")
            signal = np.zeros((target_length, signal.shape[1]), dtype=np.float32)

    current_length = signal.shape[0]
    if current_length != 5000:
        standardized_signal = np.empty((5000, signal.shape[1]), dtype=np.float32)
        if current_length < 5000:
            standardized_signal[:current_length] = signal
            standardized_signal[current_length:] = 0
        else:
            standardized_signal[:] = signal[:5000]
        signal = standardized_signal
        
    # Apply 1Hz highpass filter to suppress baseline drift
    nyquist = 0.5 * 500
    highpass_cutoff = 1 / nyquist
    b, a = butter(2, highpass_cutoff, btype='high')  # 2nd order
    signal = filtfilt(b, a, signal, axis=0)
    
    # Apply 30Hz lowpass filter to reduce high-frequency noise
    lowpass_cutoff = 30 / nyquist
    b, a = butter(2, lowpass_cutoff, btype='low')  # 2nd order
    signal = filtfilt(b, a, signal, axis=0)
    
    # Apply 50 notch filter to eliminate electrical interference
    notch_freq = 50
    bandwidth = 5
    freq = notch_freq / nyquist
    bw = bandwidth / nyquist
    b, a = butter(2, [freq - bw/2, freq + bw/2], btype='bandstop')
    signal = filtfilt(b, a, signal, axis=0)
    
    # Apply 60Hz notch filter
    notch_freq = 60
    bandwidth = 5
    freq = notch_freq / nyquist
    bw = bandwidth / nyquist
    b, a = butter(2, [freq - bw/2, freq + bw/2], btype='bandstop')
    signal = filtfilt(b, a, signal, axis=0)
        
    if np.isnan(signal).any():
        print("WARNING: Signal contains NaN values")

    signal = np.ascontiguousarray(signal.T)

    # z-score normalization with stability checks
    signal_mean = np.mean(signal, axis=0)
    signal_std = np.std(signal, axis=0)
    
    # Handle cases where std is zero or very small
    signal_std[signal_std < 1e-8] = 1.0
    
    signal = (signal - signal_mean) / signal_std
    
    # Clip extreme values
    # signal = np.clip(signal, -10, 10)
    
    # Final NaN/Inf check
    if np.isnan(signal).any() or np.isinf(signal).any():
        print("WARNING: Signal contains NaN/Inf after normalization")
        signal = np.nan_to_num(signal, nan=0.0, posinf=1e4, neginf=-1e4)

    # get meta features
    meta_features = np.empty(config.get_meta_feature_dim(), dtype=np.float32)
    ptr = 0
    
    if config.use_age:
        meta_features[ptr] = age
        ptr += 1
    if config.use_sex:
        meta_features[ptr:ptr+3] = one_hot_encoding_sex
        ptr += 3
    if config.use_signal_stats:
        valid_samples = np.isfinite(signal).sum()
        meta_features[ptr] = np.nanmean(signal) if valid_samples > 0 else 0.0
        meta_features[ptr+1] = np.nanstd(signal) if valid_samples > 1 else 0.0
        ptr += 2

    # save only the signal
    np.save(signal_path, signal.astype(np.float32))

def delete_record_files(record):
    base_name = os.path.splitext(os.path.basename(record))[0]
    signal_path = os.path.join(config.cache_folder, f'{base_name}_signal.npy')

    if os.path.exists(signal_path):
        os.remove(signal_path)

# Save your trained model.
def save_model(model_folder, state_dict):
    model_dir = os.path.join(model_folder, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save config with additional metadata
    config_dict = config.__dict__.copy()
    config_dict['meta_input_dim'] = config.get_meta_feature_dim()
    checkpoint = {
        'state_dict': state_dict,
        'config': config_dict
    }
    filename = os.path.join(model_dir, 'model.pth')
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

################################################################################
#
# ECGDataset
#
################################################################################

class ECGDataset(Dataset):
    def __init__(self, records, is_training=True):
        self.records = records
        self.is_training = is_training
        # Don't preload all paths to save memory
        self.cache_folder = config.cache_folder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        base_name = os.path.splitext(os.path.basename(self.records[idx]))[0]
        signal_path = os.path.join(self.cache_folder, f"{base_name}_signal.npy")
        
        try:
            with open(signal_path, 'rb') as f:
                signal = np.load(f)
                # Check for NaN values in signal
                if np.isnan(signal).any():
                    print(f"WARNING: Signal contains NaN values in {signal_path}")
                    signal = np.nan_to_num(signal, nan=0.0)
        except Exception as e:
            print(f"Error loading {signal_path}: {str(e)}")
            raise
        
        record = self.records[idx]
        label = float(load_label(record))
        
        # Load meta data from original record
        header = load_header(record)
        source = get_source(header)
        age = get_age(header) if config.use_age else 0
        sex = get_sex(header) if config.use_sex else 'Unknown'
        
        one_hot_encoding_sex = np.zeros(3, dtype=np.bool_)
        if sex == 'Female':
            one_hot_encoding_sex[0] = True
        elif sex == 'Male':
            one_hot_encoding_sex[1] = True
        else:
            one_hot_encoding_sex[2] = True

        # get meta features
        meta_features = np.empty(config.get_meta_feature_dim(), dtype=np.float32)
        ptr = 0
        
        if config.use_age:
            meta_features[ptr] = age
            ptr += 1
        if config.use_sex:
            meta_features[ptr:ptr+3] = one_hot_encoding_sex
            ptr += 3
        if config.use_signal_stats:
            valid_samples = np.isfinite(signal).sum()
            meta_features[ptr] = np.nanmean(signal) if valid_samples > 0 else 0.0
            meta_features[ptr+1] = np.nanstd(signal) if valid_samples > 1 else 0.0
            ptr += 2

        # Only apply data augmentation during training
        if self.is_training:
            if config.use_noise_aug and np.random.rand() < config.noise_aug_prob:
                signal = self._add_noise(signal)
                
            if config.use_scaling_aug and np.random.rand() < config.scaling_aug_prob:
                signal = self._scaling(signal)
                
            if config.use_flip_aug and np.random.rand() < config.flip_aug_prob:
                signal = np.ascontiguousarray(self._flip(signal))
                
            if config.use_shift_aug and np.random.rand() < config.shift_aug_prob:
                signal = self._shift(signal)
                
            if config.use_drop_aug and np.random.rand() < config.drop_aug_prob:
                signal = self._drop(signal)
            
            if config.add_power_noise and np.random.rand() < config.power_noise_prob:
                signal = self._add_power_noise(signal)
                
            if config.use_sine_wave_aug and np.random.rand() < config.sine_aug_prob:
                signal = self._sine_wave(signal)
                
            if config.use_square_wave_aug and np.random.rand() < config.square_aug_prob:
                signal = self._square_wave(signal)
                
            if config.use_cutout_aug and np.random.rand() < config.cutout_aug_prob:
                signal = self._cutout(signal)
                
            if config.use_time_warp_aug and np.random.rand() < config.time_wrap_prob:
                signal = self._time_wrapping(signal)
                
            if config.use_lead_mixing_aug and np.random.rand() < config.lead_mixing_prob:
                signal = self._lead_mixing_augmentation(signal, config.lead_mixing_lambda)
                
            if config.use_baseline_wander and np.random.rand() < config.baseline_wander_prob:
                signal = self._baseline_wander(signal)

            # Ensure signal is contiguous in memory and float32
            signal = np.ascontiguousarray(signal).astype(np.float32)
            meta_features = meta_features.astype(np.float32)
            
        features = [signal, meta_features]

        return features, label
        
    def _add_noise(self, signal):
        """Add Gaussian noise with zero mean and fixed standard deviation"""
        noise = np.random.normal(0, config.noise_std, signal.shape)
        augmented = signal + noise
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _add_noise")
        return augmented
        
    def _scaling(self, signal):
        """Apply random scaling to each lead between [0.5, 2.0] with numerical stability checks"""
        scaling_factors = np.random.uniform(config.scaling_min, config.scaling_max, signal.shape[0])
        scaled = signal * scaling_factors[:, np.newaxis]
        # Ensure no extreme values
        scaled = np.clip(scaled, -1e4, 1e4)
        if np.isnan(scaled).any():
            print("WARNING: NaN detected after _scaling")
        return scaled
        
    def _flip(self, signal):
        """Flip the signal vertically (up-down) by multiplying -1"""
        flipped = signal * -1
        if np.isnan(flipped).any():
            print("WARNING: NaN detected after _flip")
        return flipped
        
    def _shift(self, signal):
        """Apply cyclic shift to the signal"""
        length = signal.shape[1]
        max_shift = int(length * config.shift_max_ratio)
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        
        # Apply cyclic shift using np.roll
        shifted = np.roll(signal, shift_amount, axis=1)
        if np.isnan(shifted).any():
            print("WARNING: NaN detected after _shift")
        return shifted
        
    def _drop(self, signal):
        """Randomly drop signal points with probability [0, 0.3]"""
        mask = np.random.rand(*signal.shape) > config.drop_max_prob
        dropped = signal * mask
        if np.isnan(dropped).any():
            print("WARNING: NaN detected after _drop")
        return dropped
        
    def _sine_wave(self, signal):
        """Add sine wave with random frequency and amplitude"""
        length = signal.shape[1]
        t = np.arange(length)
        
        # Random frequency and amplitude
        freq = np.random.uniform(config.sine_min_freq, config.sine_max_freq)
        amp = np.random.uniform(0, config.sine_max_amp)
        
        # Generate sine wave
        sine = amp * np.sin(2 * np.pi * freq * t)
        
        # Add to each channel
        augmented = signal + sine[np.newaxis, :]
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _sine_wave")
        return augmented
        
    def _square_wave(self, signal):
        """Add square wave with random frequency and amplitude"""
        length = signal.shape[1]
        t = np.arange(length)
        
        # Random frequency and amplitude
        freq = np.random.uniform(config.square_min_freq, config.square_max_freq)
        amp = np.random.uniform(0, config.square_max_amp)
        
        # Generate square wave
        square = amp * np.sign(np.sin(2 * np.pi * freq * t))
        
        # Add to each channel
        augmented = signal + square[np.newaxis, :]
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _square_wave")
        return augmented
        
    def _cutout(self, signal):
        """Randomly cutout segments from random leads"""
        length = signal.shape[1]
        max_cutout = int(length * config.cutout_max_ratio)
        if max_cutout == 0:
            return signal
            
        # Randomly select leads to apply cutout (at least 1 lead)
        num_leads = signal.shape[0]
        num_cutout_leads = np.random.randint(1, num_leads + 1)
        cutout_leads = np.random.choice(num_leads, num_cutout_leads, replace=False)
        
        # Apply cutout to selected leads
        for lead in cutout_leads:
            cutout_width = np.random.randint(1, max_cutout + 1)
            start = np.random.randint(0, length - cutout_width + 1)
            signal[lead, start:start+cutout_width] = 0
            
        if np.isnan(signal).any():
            print("WARNING: NaN detected after _cutout")
        return signal
        
    def _add_power_noise(self, signal):
        """Add 50Hz power line noise as data augmentation with numerical stability checks"""
        # Calculate noise amplitude relative to signal std
        signal_std = np.std(signal)
        if signal_std > 0:
            amplitude = min(config.power_noise_amplitude * signal_std, 0.1)  # Cap amplitude
            # Generate 50Hz sine wave with random phase
            length = signal.shape[1]
            t = np.arange(length) / 500.0  # Sampling rate is 500Hz
            phase = np.random.uniform(0, 2 * np.pi)
            power_noise = amplitude * np.sin(2 * np.pi * 50 * t + phase)
            # Add to each channel with clipping
            signal = signal + power_noise
            signal = np.clip(signal, -1e4, 1e4)
            
        if np.isnan(signal).any():
            print("WARNING: NaN detected after _add_power_noise")
        return signal

    def _lead_mixing_augmentation(self, signal, lambda_val=0.2):
        """Data augmentation by mixing leads with numerical stability checks"""
        try:
            # Method 1: Pearson correlation with stability checks
            signal_std = np.std(signal, axis=1, keepdims=True)
            signal_mean = np.mean(signal, axis=1, keepdims=True)
            
            # Avoid division by zero
            signal_std[signal_std < 1e-8] = 1e-8
            normalized = (signal - signal_mean) / signal_std
            
            # Compute correlation matrix safely
            corr_matrix = np.corrcoef(normalized)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Construct adjacency matrix
            A = np.abs(corr_matrix)
            np.fill_diagonal(A, 0)
            
            # Normalize rows safely
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-8] = 1.0  # Avoid division by zero
            A = A / row_sums
            
            # Generate new signal safely
            new_signal = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                if np.any(A[i] > 0):
                    weights = A[i].reshape(-1, 1)
                    new_signal[i] = np.sum(signal * weights, axis=0)
            
            # Mix signals with clipping
            augmented = (1 - lambda_val) * signal + lambda_val * new_signal
            augmented = np.clip(augmented, -1e4, 1e4)
            
            if np.isnan(augmented).any():
                print("WARNING: NaN detected after _lead_mixing_augmentation")
            return augmented
            
        except Exception as e:
            print(f"Lead mixing failed: {str(e)}")
            return signal  # Fallback to original signal
            
    def _baseline_wander(self, signal):
        """Add baseline wander to ECG signal"""
        t = np.arange(signal.shape[1])
        freq = np.random.uniform(config.baseline_wander_min_freq, config.baseline_wander_max_freq)
        amp = config.baseline_wander_amp_ratio * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t)
        augmented = signal + drift
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _baseline_wander")
        return augmented

    def _time_wrapping(self, signal):
        """Apply time warping using numpy vectorization"""
        original_length = signal.shape[1]
        original_freq = 500  # Hz
        
        # Randomly select new frequency between min and max Hz
        new_freq = np.random.randint(config.time_wrap_min_hz, config.time_wrap_max_hz + 1)
        
        # Calculate new length after resampling
        new_length = int(original_length * new_freq / original_freq)
        
        if new_length == original_length:
            return signal.copy()
        
        # Create interpolation indices for all channels at once
        old_indices = np.linspace(0, original_length - 1, new_length)
        
        # Vectorized interpolation using numpy broadcasting
        x_old = np.arange(original_length)
        result = np.zeros((signal.shape[0], original_length))
        
        # Use numpy interp for all channels simultaneously
        for i in range(signal.shape[0]):
            resampled = np.interp(old_indices, x_old, signal[i])
            if new_length > original_length:
                result[i] = resampled[:original_length]
            else:
                result[i, :new_length] = resampled
        
        return result

################################################################################
#
# Loss Function
#
################################################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Reduced from 1.0 to 0.8 for better numerical stability
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # Ensure targets has same shape as inputs
        targets = targets.view(-1, 1).float()
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SampleWeightedLoss(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
    
    def forward(self, logits, targets):
        # Ensure targets have same shape as logits
        if len(targets.shape) < len(logits.shape):
            targets = targets.view(-1, 1)
        
        pos_weight = (1-self.beta)/(self.beta) * (targets==0).sum()/(targets==1).sum()
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight
        )

################################################################################
#
# Hybrid Model Definition
#
################################################################################

class HybridModel(nn.Module):
    def __init__(self, device, pth_path):
        super().__init__()
        # Initialize base model (without final dense layer)
        self.base_model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64,160,160,400,400,1024,1024],
            m_blocks_list=[2,2,2,3,3,4,4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=True,
            use_do=True,
            n_classes=1,  # Will be removed
            return_features=True,
            dropout_rate=config.net1d_dropout_rate
        )
        
        # Load pretrained weights (excluding dense layer)
        checkpoint = torch.load(pth_path, map_location=device)
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                     if not k.startswith('dense.')}
        self.base_model.load_state_dict(state_dict, strict=False)
        
        # Meta feature processing network with expanded dimension (256)
        meta_dim = config.get_meta_feature_dim()
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128, eps=1e-4),
            Swish(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=1e-4),
            Swish()
        )
        
        # New classification head with kaiming init and swish activation
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 256, 512),  # 1024 is the last stage output channels
            nn.BatchNorm1d(512, eps=1e-4),
            Swish(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 1)
        )
        
        # Initialize only meta_net and classifier layers with kaiming normal
        # for m in self.modules():
        for m in list(self.meta_net.modules()) + list(self.classifier.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.to(device)

    def forward(self, x, meta_features=None):
        # Get deep features directly from base model
        _, signal_features = self.base_model(x)  # Net1D now returns deep_features
        
        # Check basemodel output
        if torch.isnan(signal_features).any() or torch.isinf(signal_features).any():
            print("WARNING: BaseModel output contains NaN/Inf values")
        
        # Process meta features if provided
        if meta_features is not None:
            if meta_features.dim() == 1:  # Handle single sample case
                meta_features = meta_features.unsqueeze(0)
            meta_features = self.meta_net(meta_features)
            
            # Check meta_net output
            if torch.isnan(meta_features).any() or torch.isinf(meta_features).any():
                print("WARNING: MetaNet output contains NaN/Inf values")
            
            features = torch.cat([signal_features, meta_features], dim=1)
        else:
            features = signal_features
            
        # Get model output with NaN check
        output = self.classifier(features)
        
        # Check classifier output before final check
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: Classifier output contains NaN/Inf values")
            
        return output
