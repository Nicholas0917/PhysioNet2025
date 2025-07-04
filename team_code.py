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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import psutil

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
        self.model_name = 'ecgfounder'
        self.use_pretrained = True
        self.pretrain_num_epochs = 50
        self.pretrain_learning_rate = 2e-5
        self.pretrain_batch_size = 256
        self.gradient_accumulation_steps = 4
        self.pretrain_early_stop_patience = 5
        self.num_epochs = 100
        self.learning_rate = 1e-6
        self.dropout_rate = 0.3
        self.net1d_dropout_rate = 0.3
        self.batch_size = 32
        self.early_stop_patience = 8
        self.num_preprocess_workers = 2
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_folder = '/mnt/scratch/wmqn2362/PhysioNet25/tmp'

        # Loss parameters
        # self.pretrain_focal_alpha = 0.6
        self.pretrain_focal_gamma = 2
        self.pretrain_margin = 0.2
        self.pretrain_s = 30
        self.pretrain_lmf_alpha = 0.02
        self.pretrain_lmf_beta = 0.98
        # self.finetune_focal_alpha = 0.8
        self.finetune_focal_gamma = 2
        self.finetune_margin = 0.35
        self.finetune_s = 30
        self.finetune_lmf_alpha = 0.02
        self.finetune_lmf_beta = 0.98
        
        # WeightedRandomSampler parameters
        self.pos_sample_weight_multiplier = 1.0
        
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
        
        self.use_drop_aug = True
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
        
        self.use_time_warp_aug = True
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

        print(">>>>>>>>>Loss Parameters:<<<<<<<<<<")
        # print(f"Pretrain Focal Alpha: {self.pretrain_focal_alpha}")
        print(f"Pretrain Focal Gamma: {self.pretrain_focal_gamma}")
        print(f"Pretrain Margin: {self.pretrain_margin}")
        print(f"Pretrain S: {self.pretrain_s}")
        print(f"Pretrain LMF Alpha: {self.pretrain_lmf_alpha}")
        print(f"Pretrain LMF Beta: {self.pretrain_lmf_beta}")
        # print(f"Finetune Focal Alpha: {self.finetune_focal_alpha}")
        print(f"Finetune Focal Gamma: {self.finetune_focal_gamma}")
        print(f"Finetune Margin: {self.finetune_margin}")
        print(f"Finetune S: {self.finetune_s}")
        print(f"Finetune LMF Alpha: {self.finetune_lmf_alpha}")
        print(f"Finetune LMF Beta: {self.finetune_lmf_beta}")
        print(f"Positive Sample Weight Multiplier: {self.pos_sample_weight_multiplier}")
        
        print(">>>>>>>>>Device:<<<<<<<<<<")
        print(f"Device: {self.device}")

config = Config()
config.print_config()

# Configure CUDA/cuDNN for stability
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True

# os.environ['CUDNN_V8_API_ENABLED'] = '0'

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
    """Train the model using the three-stage process"""
    torch.autograd.set_detect_anomaly(True)
    
    ############################################################################
    # Stage 0: Data Loading and Preprocessing
    ############################################################################
    records = find_records(data_folder)
    with ThreadPoolExecutor(max_workers=config.num_preprocess_workers) as executor:
        list(executor.map(lambda r: data_preprocess(r, config), records))
    
    # Split into CODE-15% and other records (parallel processing)
    def classify_record(record):
        header = load_header(os.path.join(data_folder, record))
        source = get_source(header)
        return (record, source)
    
    with ThreadPoolExecutor(max_workers=config.num_preprocess_workers) as executor:
        results = list(executor.map(classify_record, records))
    
    code15_records = [os.path.join(data_folder, r) for r, src in results if src == 'CODE-15%']
    finetune_records = [os.path.join(data_folder, r) for r, src in results if src != 'CODE-15%']
    
    # Create datasets
    pretrain_dataset = ECGDataset(code15_records, is_training=True, config=config)
    finetune_dataset = ECGDataset(finetune_records, is_training=True, config=config)
    
    ############################################################################
    # Stage 1: Pretrain Model
    ############################################################################
    # Initialize model and training components
    model = HybridModel(
        device=config.device,
        pth_path='./12_lead_ECGFounder.pth',
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
    
    # # Use WeightBCE as loss function
    # # Calculate pos_weight based on initial probability and multiplier
    # total_samples = pretrain_pos_samples + pretrain_neg_samples
    # initial_pos_ratio = pretrain_pos_samples / total_samples if total_samples > 0 else 0.0
    
    # # Ensure initial_pos_ratio is not zero to avoid division by zero
    # if initial_pos_ratio == 0:
    #     pos_weight_value = config.pos_sample_weight_multiplier # Fallback if no positive samples
    # else:
    #     pos_weight_value = (1.0 - initial_pos_ratio) / initial_pos_ratio * config.pos_sample_weight_multiplier
    
    # pos_weight_tensor = torch.tensor([pos_weight_value], device=config.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    
    pretrain_pos_samples_weighted = pretrain_pos_samples * config.pos_sample_weight_multiplier
    criterion = create_binary_lmf_loss(
        pos_samples=pretrain_pos_samples_weighted,
        neg_samples=pretrain_neg_samples,
        device=config.device,
        alpha=config.pretrain_lmf_alpha,
        beta=config.pretrain_lmf_beta,
        focal_gamma=config.pretrain_focal_gamma,
        ldam_margin=config.pretrain_margin,
        ldam_s=config.pretrain_s
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.pretrain_learning_rate,
        weight_decay=2e-4
    )
    
    # Warmup scheduler
    warmup_epochs = int(config.pretrain_num_epochs * 0.1)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + (epoch / warmup_epochs) * 0.9
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (config.pretrain_num_epochs - warmup_epochs)))
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_num_epochs - warmup_epochs,
        eta_min=config.pretrain_learning_rate * 0.01
    )
    
    model = pretrain_model(
        pretrain_dataset=pretrain_dataset,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        warmup_scheduler=warmup_scheduler,
        scheduler=scheduler,
        num_epochs=config.pretrain_num_epochs,
        batch_size=config.pretrain_batch_size,
        early_stop_patience=config.pretrain_early_stop_patience,
        device=config.device,
        pretrain_model_pth=os.path.join(model_folder, 'pretrain_model.pth'),
        verbose=verbose,
    )
    
    ############################################################################
    # Stage 2: Evaluate pretrained model
    ############################################################################
    pretrain_eval_dataset = ECGDataset(code15_records, is_training=False, config=config)
    finetune_eval_dataset = ECGDataset(finetune_records, is_training=False, config=config)
    evaluate_model(model, pretrain_eval_dataset, finetune_eval_dataset, verbose)
    
    ############################################################################
    # Stage 3: Finetune on target datasets
    ############################################################################
    # Freeze all layers except classifier in the pretrained model
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Print model parameters after freezing layers for finetuning
    print_model_parameters(model, verbose)

    # Initialize training components with BinaryLMFLoss
    # For finetuning, we also use BinaryLMFLoss
    finetune_labels = [finetune_dataset[i][1] for i in range(len(finetune_dataset))]
    finetune_neg_samples = finetune_labels.count(0)
    finetune_pos_samples = finetune_labels.count(1)

    # # Use WeightBCE as loss function for finetuning
    # # Calculate pos_weight based on initial probability and multiplier for finetuning
    # total_finetune_samples = finetune_pos_samples + finetune_neg_samples
    # initial_finetune_pos_ratio = finetune_pos_samples / total_finetune_samples if total_finetune_samples > 0 else 0.0

    # if initial_finetune_pos_ratio == 0:
    #     finetune_pos_weight_value = config.pos_sample_weight_multiplier # Fallback if no positive samples
    # else:
    #     finetune_pos_weight_value = (1.0 - initial_finetune_pos_ratio) / initial_finetune_pos_ratio * config.pos_sample_weight_multiplier
    
    # pos_weight_tensor = torch.tensor([finetune_pos_weight_value], device=config.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    finetune_pos_samples_weighted = finetune_pos_samples * config.pos_sample_weight_multiplier
    criterion = create_binary_lmf_loss(
        pos_samples=finetune_pos_samples_weighted,
        neg_samples=finetune_neg_samples,
        device=config.device,
        alpha=config.finetune_lmf_alpha,
        beta=config.finetune_lmf_beta,
        focal_gamma=config.finetune_focal_gamma,
        ldam_margin=config.finetune_margin,
        ldam_s=config.finetune_s
    )
    
    # Create optimizers and schedulers for each fold
    optimizers = []
    schedulers = []
    for _ in range(5):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # scaler = torch.amp.GradScaler('cuda')  # Disabled mixed precision training
    # autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)  # Disabled mixed precision training
    kf = StratifiedKFold(n_splits=5)
    
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
    
    if verbose:
        print('Done.')
        print()

def pretrain_model(pretrain_dataset, model, criterion, optimizer, warmup_epochs, warmup_scheduler, scheduler,
                  num_epochs, batch_size, early_stop_patience, device,
                  pretrain_model_pth, verbose):
    """Core pretraining logic with training loop and model saving"""
    torch.autograd.set_detect_anomaly(True)
    
    # Check if pretrained model exists
    if os.path.exists(pretrain_model_pth):
        if verbose:
            print(f'Loading pretrained model from {pretrain_model_pth}')
        try:
            checkpoint = torch.load(pretrain_model_pth, map_location=device)
            # Handle both full checkpoint and state_dict cases
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if verbose:
                print('Successfully loaded pretrained model')
            return model
        except Exception as e:
            if verbose:
                print(f'Failed to load pretrained model: {str(e)}')
            print('Proceeding with training from scratch')
    
    if verbose:
        print(f'Starting pretraining on {len(pretrain_dataset)} records...')
    
    # scaler = torch.amp.GradScaler('cuda')  # Disabled mixed precision training
    # autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)  # Disabled mixed precision training

    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(pretrain_dataset))
    val_size = len(pretrain_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])

    # Create weighted sampler for imbalanced data (commented out as requested)
    train_weights = make_weights_for_balanced_classes(train_dataset)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            num_workers=config.num_preprocess_workers)
    val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=config.num_preprocess_workers)

    best_auprc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []
        pos_logits_sum = 0.0
        neg_logits_sum = 0.0
        pos_probs_sum = 0.0
        pos_count = 0
        neg_count = 0
        
        for i, (features, label) in enumerate(train_loader):
            signal, meta_features = features
            
            signal = signal.to(device)
            meta_features = meta_features.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # with autocast:
            # Check input data
            if torch.isnan(signal).any() or torch.isinf(signal).any():
                print(f"WARNING: Input signal contains NaN/Inf values at batch {i}")
            if torch.isnan(meta_features).any() or torch.isinf(meta_features).any():
                print(f"WARNING: Meta features contain NaN/Inf values at batch {i}")
            
            output = model(signal, meta_features)
            
            # Check model output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: Model output contains NaN/Inf values at batch {i}")
                print(f"Output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
            
            # Reshape label to match output shape [batch_size, 1]
            label_reshaped = label.view(-1, 1)
            loss = criterion(output, label_reshaped)
            
            # Check loss value
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"WARNING: Loss contains NaN/Inf values at batch {i}")
            
            # Gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            # scaler.scale(loss).backward()
            loss.backward()
            check_gradients(model)
            
            if (i + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                optimizer.zero_grad()
            
                train_loss += loss.item() * config.gradient_accumulation_steps
                train_targets.extend(label.cpu().numpy())
                train_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())
                
                # Calculate positive/negative logit stats
                pos_mask = label == 1
                neg_mask = ~pos_mask
                if pos_mask.any():
                    pos_logits = output[pos_mask].mean().item()
                    pos_probs = torch.sigmoid(output[pos_mask]).mean().item()
                else:
                    pos_logits = 0.0
                    pos_probs = 0.0
                if neg_mask.any():
                    neg_logits = output[neg_mask].mean().item()
                else:
                    neg_logits = 0.0
                
                # Accumulate stats for epoch average
                if pos_mask.any():
                    pos_logits_sum += pos_logits * pos_mask.sum().item()
                    pos_probs_sum += pos_probs * pos_mask.sum().item()
                    pos_count += pos_mask.sum().item()
                if neg_mask.any():
                    neg_logits_sum += neg_logits * neg_mask.sum().item()
                    neg_count += neg_mask.sum().item()

        train_loss /= len(train_loader)
        
        # Calculate epoch averages
        epoch_pos_logit = pos_logits_sum / pos_count if pos_count > 0 else 0.0
        epoch_neg_logit = neg_logits_sum / neg_count if neg_count > 0 else 0.0
        epoch_pos_prob = pos_probs_sum / pos_count if pos_count > 0 else 0.0
        
        print(f"Epoch {epoch + 1} Averages:")
        print(f"  Positive Logit: {epoch_pos_logit:.4f}")
        print(f"  Negative Logit: {epoch_neg_logit:.4f}") 
        print(f"  Positive Probability: {epoch_pos_prob:.4f}")
        
        # Calculate training metrics
        train_auroc = roc_auc_score(train_targets, np.round(train_outputs))
        train_auprc = average_precision_score(train_targets, np.round(train_outputs))
        train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
        train_f1 = f1_score(train_targets, np.round(train_outputs))

        # Validation
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []
        val_pos_logits_sum = 0.0
        val_neg_logits_sum = 0.0
        val_pos_probs_sum = 0.0
        val_pos_count = 0
        val_neg_count = 0
        
        with torch.no_grad():
            for features, label in val_loader:
                signal, meta_features = features
                signal = signal.to(device)
                meta_features = meta_features.to(device)
                label = label.to(device)

                # with autocast:
                output = model(signal, meta_features)
                # Reshape label to match output shape [batch_size, 1]
                label_reshaped = label.view(-1, 1)
                loss = criterion(output, label_reshaped)
                
                val_loss += loss.item()
                val_targets.extend(label.cpu().numpy())
                val_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())
                
                # Calculate positive/negative logit stats for validation
                pos_mask = label == 1
                neg_mask = ~pos_mask
                if pos_mask.any():
                    pos_logits = output[pos_mask].mean().item()
                    pos_probs = torch.sigmoid(output[pos_mask]).mean().item()
                    val_pos_logits_sum += pos_logits * pos_mask.sum().item()
                    val_pos_probs_sum += pos_probs * pos_mask.sum().item()
                    val_pos_count += pos_mask.sum().item()
                if neg_mask.any():
                    neg_logits = output[neg_mask].mean().item()
                    val_neg_logits_sum += neg_logits * neg_mask.sum().item()
                    val_neg_count += neg_mask.sum().item()

        val_loss /= len(val_loader)
        
        # Calculate validation epoch averages
        val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
        val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
        val_epoch_pos_prob = val_pos_probs_sum / val_pos_count if val_pos_count > 0 else 0.0
        
        print(f"Validation Averages:")
        print(f"  Positive Logit: {val_epoch_pos_logit:.4f}")
        print(f"  Negative Logit: {val_epoch_neg_logit:.4f}")
        print(f"  Positive Probability: {val_epoch_pos_prob:.4f}")
        
        # Calculate validation metrics
        val_outputs_arr = np.array(val_outputs)
        val_targets_arr = np.array(val_targets)

        if len(np.unique(val_targets_arr)) < 2:
            print("WARNING: Only one class present in validation targets")
            val_auroc = 0.5
        else:
            val_auroc = roc_auc_score(val_targets_arr, val_outputs_arr)
        
        val_auprc = average_precision_score(val_targets_arr, val_outputs_arr)
        val_accuracy = accuracy_score(val_targets_arr, np.round(val_outputs_arr))
        val_f1 = f1_score(val_targets_arr, np.round(val_outputs_arr))
            
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        epoch_duration = time.time() - epoch_start_time

        # Print training progress
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
        print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
        print(f'Valid AUROC: {val_auroc:.4f}, Valid AUPRC: {val_auprc:.4f}, Valid Accuracy: {val_accuracy:.4f}, Valid F1: {val_f1:.4f}\n')

        # Early stopping based on AUPRC
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping: Valid AUPRC not improved for {early_stop_patience} epochs")
                break

    # Save final model
    os.makedirs(os.path.dirname(pretrain_model_pth), exist_ok=True)
    torch.save(best_model, pretrain_model_pth)
    if verbose:
        print('Pretraining completed and model saved')
    
    # Return the trained model
    model.load_state_dict(best_model)
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
    torch.autograd.set_detect_anomaly(True)
    
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
            pos_logits_sum = 0.0
            neg_logits_sum = 0.0
            pos_probs_sum = 0.0
            pos_count = 0
            neg_count = 0
            
            for i, (features, label) in enumerate(train_loader):
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                label = label.to(config.device)

                optimizers[fold].zero_grad()
                # with autocast:  # Disabled mixed precision training
                # Check input data
                if torch.isnan(signal).any() or torch.isinf(signal).any():
                    print(f"WARNING: Input signal contains NaN/Inf values at batch {i}")
                if torch.isnan(meta_features).any() or torch.isinf(meta_features).any():
                    print(f"WARNING: Meta features contain NaN/Inf values at batch {i}")
                
                output = model(signal, meta_features)
                
                # Check model output
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"WARNING: Model output contains NaN/Inf values at batch {i}")
                    print(f"Output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
                
                # Reshape label to match output shape [batch_size, 1]
                label_reshaped = label.view(-1, 1)
                loss = criterion(output, label_reshaped)
                
                # Check loss value
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: Loss contains NaN/Inf values at batch {i}")
                
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
                train_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())

                # Calculate positive/negative logit stats
                pos_mask = label == 1
                neg_mask = ~pos_mask
                if pos_mask.any():
                    pos_logits = output[pos_mask].mean().item()
                    pos_probs = torch.sigmoid(output[pos_mask]).mean().item()
                else:
                    pos_logits = 0.0
                    pos_probs = 0.0
                if neg_mask.any():
                    neg_logits = output[neg_mask].mean().item()
                else:
                    neg_logits = 0.0
                
                # Accumulate stats for epoch average
                if pos_mask.any():
                    pos_logits_sum += pos_logits * pos_mask.sum().item()
                    pos_probs_sum += pos_probs * pos_mask.sum().item()
                    pos_count += pos_mask.sum().item()
                if neg_mask.any():
                    neg_logits_sum += neg_logits * neg_mask.sum().item()
                    neg_count += neg_mask.sum().item()

            train_loss /= len(train_loader)
            
            # Calculate epoch averages
            epoch_pos_logit = pos_logits_sum / pos_count if pos_count > 0 else 0.0
            epoch_neg_logit = neg_logits_sum / neg_count if neg_count > 0 else 0.0
            epoch_pos_prob = pos_probs_sum / pos_count if pos_count > 0 else 0.0
            
            print(f"Epoch {epoch + 1} Averages:")
            print(f"  Positive Logit: {epoch_pos_logit:.4f}")
            print(f"  Negative Logit: {epoch_neg_logit:.4f}") 
            print(f"  Positive Probability: {epoch_pos_prob:.4f}")

            schedulers[fold].step()

            # Calculate training metrics
            train_auroc = roc_auc_score(train_targets, np.round(train_outputs))
            train_auprc = average_precision_score(train_targets, np.round(train_outputs))
            train_accuracy = accuracy_score(train_targets, np.round(train_outputs))
            train_f1 = f1_score(train_targets, np.round(train_outputs))

            # Validation
            model.eval()
            val_loss = 0.0
            val_targets = []
            val_outputs = []
            val_pos_logits_sum = 0.0
            val_neg_logits_sum = 0.0
            val_pos_probs_sum = 0.0
            val_pos_count = 0
            val_neg_count = 0
            with torch.no_grad():
                for i, (features, label) in enumerate(val_loader):
                    signal, meta_features = features
                    signal = signal.to(config.device)
                    meta_features = meta_features.to(config.device)
                    label = label.to(config.device)

                    # with autocast:
                    output = model(signal, meta_features)
                    # Reshape label to match output shape [batch_size, 1]
                    label_reshaped = label.view(-1, 1)
                    loss = criterion(output, label_reshaped)
                    
                    val_loss += loss.item()
                    val_targets.extend(label.cpu().numpy())
                    val_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())

                    # Calculate positive/negative logit stats for validation
                    pos_mask = label == 1
                    neg_mask = ~pos_mask
                    if pos_mask.any():
                        pos_logits = output[pos_mask].mean().item()
                        pos_probs = torch.sigmoid(output[pos_mask]).mean().item()
                        val_pos_logits_sum += pos_logits * pos_mask.sum().item()
                        val_pos_probs_sum += pos_probs * pos_mask.sum().item()
                        val_pos_count += pos_mask.sum().item()
                    if neg_mask.any():
                        neg_logits = output[neg_mask].mean().item()
                        val_neg_logits_sum += neg_logits * neg_mask.sum().item()
                        val_neg_count += neg_mask.sum().item()

            val_loss /= len(val_loader)
            
            # Calculate validation epoch averages
            val_epoch_pos_logit = val_pos_logits_sum / val_pos_count if val_pos_count > 0 else 0.0
            val_epoch_neg_logit = val_neg_logits_sum / val_neg_count if val_neg_count > 0 else 0.0
            val_epoch_pos_prob = val_pos_probs_sum / val_pos_count if val_pos_count > 0 else 0.0
            
            print(f"Validation Averages:")
            print(f"  Positive Logit: {val_epoch_pos_logit:.4f}")
            print(f"  Negative Logit: {val_epoch_neg_logit:.4f}")
            print(f"  Positive Probability: {val_epoch_pos_prob:.4f}")

            val_auroc = roc_auc_score(val_targets, np.round(val_outputs))
            val_auprc = average_precision_score(val_targets, np.round(val_outputs))
            val_accuracy = accuracy_score(val_targets, np.round(val_outputs))
            val_f1 = f1_score(val_targets, np.round(val_outputs))

            epoch_duration = time.time() - epoch_start_time
            
            if verbose:
                print(f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
                print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
                print(f'Valid AUROC: {val_auroc:.4f}, Valid AUPRC: {val_auprc:.4f}, Valid Accuracy: {val_accuracy:.4f}, Valid F1: {val_f1:.4f}\n')

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
        
        end_time = time.time()
        if verbose:
            print(f'Fold {fold + 1} finished. Best Valid AUPRC: {best_auprc:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

        # Save finetuned model
        os.makedirs(model_folder, exist_ok=True)
        save_model(model_folder, best_model, config, fold=fold+1)
        if verbose:
            print(f'Finetuning completed for fold {fold+1}')

def evaluate_model(model, pretrain_dataset, finetune_dataset, verbose):
    """Evaluate model performance on both pretrain and finetune datasets"""
    model.eval()
    
    def evaluate_dataset(dataset, dataset_name):
        loader = DataLoader(dataset,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_preprocess_workers)
        
        # Evaluation
        outputs = []
        targets = []
        with torch.no_grad():
            for features, label in loader:
                signal, meta_features = features
                signal = signal.to(config.device)
                meta_features = meta_features.to(config.device)
                label = label.to(config.device)

                output = model(signal, meta_features)
                outputs.extend(torch.sigmoid(output).cpu().numpy())
                targets.extend(label.cpu().numpy())
        
        # Calculate metrics
        auroc = roc_auc_score(targets, outputs)
        auprc = average_precision_score(targets, outputs)
        accuracy = accuracy_score(targets, np.round(outputs))
        f1 = f1_score(targets, np.round(outputs))
        
        if verbose:
            print(f"\n{dataset_name} Dataset Metrics:")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
    
    # Evaluate on pretrain dataset
    evaluate_dataset(pretrain_dataset, "Pretrain")
    
    # Evaluate on finetune dataset
    evaluate_dataset(finetune_dataset, "Finetune")

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
    data_preprocess(record, config)

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

    # Handle single model or model list
    if isinstance(model, list):
        # Ensemble prediction
        probabilities = []
        for m in model:
            m.eval()
            with torch.no_grad():
                output = m(signal, meta_features)
                prob = torch.sigmoid(output.view(-1)[0]).item()
                probabilities.append(prob)
        probability_output = sum(probabilities) / len(probabilities)
    else:
        # Single model prediction
        model.eval()
        with torch.no_grad():
            output = model(signal, meta_features)
            probability_output = torch.sigmoid(output.view(-1)[0]).item()

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
    return weights
