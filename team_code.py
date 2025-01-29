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
from torch.utils.data import DataLoader, Dataset, Subset
import math
import time
from sklearn.model_selection import KFold

from helper_code import *

################################################################################
#
# Global configuration. 
#
################################################################################

class Config:
    def __init__(self):
        self.model_name = 'resnet50'
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.dropout_rate = 0.2
        self.batch_size = 32
        self.early_stop_patience = 5
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_pretrained = False

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
        print(">>>>>>>>>Training Parameters:<<<<<<<<<<")
        print(f"Model Name: {self.model_name}")
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
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
    for i in range(num_records):
        records[i] = os.path.join(data_folder, records[i])

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # get data using data_loader
    dataset = ECGDataset(records)

    ############################################################################
    # Train the models.
    if verbose:
        print('Training the model on the data...')

    # Define the parameters using config.
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    early_stop_patience = config.early_stop_patience
    device = config.device

    # Define the model.
    if config.model_name == 'resnet18':
        model = resnet18(pretrained=config.use_pretrained).to(device)
    elif config.model_name == 'resnet34':
        model = resnet34(pretrained=config.use_pretrained).to(device)
    elif config.model_name == 'resnet50':
        model = resnet50(pretrained=config.use_pretrained).to(device)
    elif config.model_name == 'resnet101':
        model = resnet101(pretrained=config.use_pretrained).to(device)
    elif config.model_name == 'resnet152':
        model = resnet152(pretrained=config.use_pretrained).to(device)
    else:
        raise ValueError('Invalid model name.')

    # Fit the model.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')
    kf = KFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        best_loss = float('inf')
        best_epoch = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
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

            train_loss /= len(train_loader)
            scheduler.step()

            model.eval()
            val_loss = 0.0
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

            val_loss /= len(val_loader)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = model.state_dict()
            else:
                if epoch - best_epoch > early_stop_patience:
                    break

        end_time = time.time()
        print(f'Fold {fold + 1} finished. Best Val Loss: {best_loss:.4f} at epoch {best_epoch + 1}. Time: {end_time - start_time:.2f} seconds')

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
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model_state_dict = model['model']
    
    # Define the model architecture
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
        raise ValueError('Invalid model name.')
    
    # Load the state dictionary into the model
    model.load_state_dict(model_state_dict)
    model.eval()

    # Extract the features.
    signal, meta_features = extract_features(record)

    # extend batch dimension
    signal = np.expand_dims(signal, axis=0)
    meta_features = np.expand_dims(meta_features, axis=0)
    
    # transfer to device
    signal = torch.from_numpy(signal).to(config.device)
    meta_features = torch.from_numpy(meta_features).to(config.device)    

    # Get the model outputs.
    probability_output = model(signal, meta_features)
    probability_output = torch.sigmoid(probability_output).detach().cpu().numpy().item()
    binary_output = probability_output > 0.5

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header) if config.use_age else 0
    sex = get_sex(header) if config.use_sex else 'Unknown'
    
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    if signal.shape[0] < 4096:
        signal = np.pad(signal, ((0, 4096 - signal.shape[0]), (0, 0)), 'constant')

    meta_features = []
    if config.use_age:
        meta_features.append(age)
    if config.use_sex:
        meta_features.extend(one_hot_encoding_sex)
    if config.use_signal_stats:
        meta_features.extend([signal_mean, signal_std])

    signal = signal.T

    return [np.asarray(signal, dtype=np.float32), np.asarray(meta_features, dtype=np.float32)]

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

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
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc(x).squeeze()
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