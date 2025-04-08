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
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from helper_code import *

################################################################################
#
# Global configuration. 
#
################################################################################

class Config:
    def __init__(self):
        self.model_name = 'resnet50'
        self.use_pretrained = True
        self.pretrain_num_epochs = 100
        self.pretrain_learning_rate = 1e-4
        self.pretrain_batch_size = 128
        self.pretrain_early_stop_patience = 3
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.dropout_rate = 0.25
        self.batch_size = 32
        self.early_stop_patience = 3
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(f"Batch Size: {self.batch_size}")
        print(f"Early Stop Patience: {self.early_stop_patience}")

        print(">>>>>>>>>Meta Features:<<<<<<<<<<")
        print(f"Use Age: {self.use_age}")
        print(f"Use Sex: {self.use_sex}")
        print(f"Use Signal Stats: {self.use_signal_stats}")
        print(f"Meta Feature Dimension: {self.get_meta_feature_dim()}")

        print(">>>>>>>>>Device:<<<<<<<<<<")
        print(f"Device: {self.device}")

config = Config()
config.print_config()

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):

    ############################################################################
    # Load the data.
    records = find_records(data_folder)
    num_records = len(records)
    
    print(f'Total number of records: {num_records}')
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # divide the records according to the source
    code15_records = []
    PTBXL_records = []
    SaMiTrop_records = []

    for record in records:
        record_path = os.path.join(data_folder, record)
        header = load_header(record_path)

        if get_source(header) == 'PTB-XL':
            PTBXL_records.append(record_path)
        elif get_source(header) == 'CODE-15%':
            code15_records.append(record_path)
        elif get_source(header) == 'SaMi-Trop':
            SaMiTrop_records.append(record_path)
        else:
            raise ValueError('Invalid source.')
      
    Code15_records_pretrain = code15_records
    Code15_records_finetune = []
      
    # Pretrain
    if verbose:
        print('Pretraining the model on the CODE%15 data...')

    print("Pretrain Datastes Size: ",len(Code15_records_pretrain))
    dataset = ECGDataset(Code15_records_pretrain) 

    ############################################################################
    # Pretrain the models.

    num_epochs = config.pretrain_num_epochs
    learning_rate = config.pretrain_learning_rate
    batch_size = config.pretrain_batch_size
    early_stop_patience = config.pretrain_early_stop_patience
    device = config.device

    # Define the model.
    if config.model_name == 'resnet18':
        model = resnet18().to(device)
    elif config.model_name == 'resnet34':
        model = resnet34().to(device)
    elif config.model_name == 'resnet50':
        model = resnet50().to(device)
    elif config.model_name == 'resnet101':
        model = resnet101().to(device)
    elif config.model_name == 'resnet152':
        model = resnet152().to(device)
    else:
        raise ValueError('Invalid model name.')

    # Fit the model.
    criterion = FocalLoss(alpha=0.8, logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')
    kf = StratifiedKFold(n_splits=5)

    def make_weights_for_balanced_classes(dataset):
        targets = [dataset[i][1] for i in range(len(dataset))]
        weights = np.zeros_like(targets, dtype=np.float32)
        weights[np.isclose(targets, 0.0)] = 1.0
        weights[np.isclose(targets, 1.0)] = 10.0
        return weights

    X = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=6)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=6)

        best_loss = float('inf')
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
                with torch.amp.autocast('cuda'):
                    output = model(signal, meta_features)
                    loss = criterion(output, label)
                scaler.scale(loss).backward()
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

                    with torch.amp.autocast('cuda'):
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

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
            print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
            print(f'Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}\n')

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = model.state_dict()
            else:
                if epoch - best_epoch > early_stop_patience:
                    break

        end_time = time.time()
        print(f'Fold {fold + 1} finished. Best Val Loss: {best_loss:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')

    pretrained_model = model.state_dict().copy()
    
    ############################################################################
    # fine-tune stage
    if verbose:
        print('Training the model on the fine-tune data...\n')
        
    finetune_records = PTBXL_records + SaMiTrop_records + Code15_records_finetune
    print("Fine-tune Datastes Size: ",len(finetune_records))
    dataset = ECGDataset(finetune_records)

    ############################################################################
    # Train the models.
    # Define the parameters using config.
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    early_stop_patience = config.early_stop_patience
    device = config.device

    # Fit the model.
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=0.8, logits=True)
    # kf = KFold(n_splits=3)
    kf = StratifiedKFold(n_splits=5)

    X = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    finetuned_models = []

    # for fold, (train_idx, val_idx) in enumerate(kf.split(records)):
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_weights = make_weights_for_balanced_classes(train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=6)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=6)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=6)

        best_loss = float('inf')
        best_epoch = 0

        # model = pretrained_model
        if config.model_name == 'resnet18':
            model = resnet18().to(device)
        elif config.model_name == 'resnet34':
            model = resnet34().to(device)
        elif config.model_name == 'resnet50':
            model = resnet50().to(device)
        elif config.model_name == 'resnet101':
            model = resnet101().to(device)
        elif config.model_name == 'resnet152':
            model = resnet152().to(device)
        else:
            raise ValueError('Invalid model name.')
        model.load_state_dict(pretrained_model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scaler = torch.amp.GradScaler('cuda')

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
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(signal, meta_features)
                    loss = criterion(output, label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
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

                    with torch.amp.autocast('cuda'):
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
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
            print(f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
            print(f'Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}\n')

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = model.state_dict()
            else:
                if epoch - best_epoch > early_stop_patience:
                    break
    
        end_time = time.time()
        print(f'Fold {fold + 1} finished. Best Val Loss: {best_loss:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds \n')
        finetuned_models.append(best_model)

    save_model(model_folder=model_folder, models=finetuned_models)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):

    model_path = os.path.join(model_folder, 'ensemble_models.pth')
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=True)
    
    model_keys = sorted(checkpoint.keys(), key=lambda x: int(x[5:]))
    models_state = [checkpoint[key] for key in model_keys]
    
    models = []
    for state in models_state:
        if config.model_name == 'resnet18':
            model = resnet18().to(config.device)
        elif config.model_name == 'resnet34':
            model = resnet34().to(config.device)
        elif config.model_name == 'resnet50':
            model = resnet50().to(config.device)
        elif config.model_name == 'resnet101':
            model = resnet101().to(config.device)
        elif config.model_name == 'resnet152':
            model = resnet152().to(config.device)
        else:
            raise ValueError(f'Unsupported model: {config.model_name}')

        model.load_state_dict(state)
        model.eval()
        models.append(model)
    
    if verbose:
        print(f'Successfully loaded {len(models)} models from: {model_folder}')
    
    return models

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):

    models = model

    signal, meta_features = extract_features(record)
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(config.device)
    meta_tensor = torch.FloatTensor(meta_features).unsqueeze(0).to(config.device)

    all_probs = []
    with torch.no_grad():
        for model in models:
            output = model(signal_tensor, meta_tensor)
            prob = torch.sigmoid(output).detach().cpu().numpy().item()
            all_probs.append(prob)

    avg_prob = np.mean(all_probs)
    return int(avg_prob > 0.5), float(avg_prob)

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# helper function to load the source
def get_source(string):
    source_string = '# Source:'
    source, has_source = get_variable(string, source_string)
    return source

# Extract your features.
def extract_features(record):
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
    signal = signal.astype(np.float32)

    # transfer fs
    if source == 'PTB-XL':
        original_fs = 500
        target_fs = 400
        target_length = int(signal.shape[0] * target_fs / original_fs)
        # resampled_signal = np.empty((target_length, signal.shape[1]), dtype=np.float32)
        resampled_signal = resample(signal, target_length, axis=0).astype(np.float32)
        signal = resampled_signal

    current_length = signal.shape[0]
    if current_length != 4096:
        standardized_signal = np.empty((4096, signal.shape[1]), dtype=np.float32)
        if current_length < 4096:
            standardized_signal[:current_length] = signal
            standardized_signal[current_length:] = 0
        else:
            standardized_signal[:] = signal[:4096]
        signal = standardized_signal
    
    # filter the signal using a 0.5hz - 40hz bandpass filter
    nyquist = 0.5 * 400
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = butter(4, [low, high], btype='band')  # 4-order
    signal = filtfilt(b, a, signal, axis=0)
    
    if np.isnan(signal).any():
        np.nan_to_num(signal, copy=False)

    # normalize the signal
    # min-max normalization
    # signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0) + 1e-8)
    # z-score normalization
    signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

    signal = signal.astype(np.float32)
    signal = np.ascontiguousarray(signal.T)

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

    return [signal, meta_features]

# Save your trained model.
def save_model(model_folder, models):
    
    os.makedirs(model_folder, exist_ok=True)
    save_path = os.path.join(model_folder, 'ensemble_models.pth')
    
    model_dict = {}
    for i, model in enumerate(models, start=1):
        model_dict[f'model{i}'] = model
    
    torch.save(model_dict, save_path)
    print(f'Ensemble models saved to {save_path}\n')

################################################################################
#
# Dataset
#
################################################################################

class ECGDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        features = extract_features(record)
        label = float(load_label(record))

        return features, label

################################################################################
#
# Loss Function
#
################################################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
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

################################################################################
#
# SEResNet
#
################################################################################

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se = SELayer(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=12, out_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(config.get_meta_feature_dim(), 32)
        self.fc = nn.Linear(512 * block.expansion + 32, out_channel)
        self.dropout = nn.Dropout(config.dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ag):
        # x.shape = [batch_size, 12, 4096], ag.shape = [batch_size, n]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        x = self.dropout(x)
        x = self.fc(x).squeeze(1)
        return x
    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model