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
from sklearn.mixture import GaussianMixture
from helper_code import *
import torch.cuda.amp as amp

################################################################################
#
# Global configuration. 
#
################################################################################

class Config:
    def __init__(self):
        self.model_name = 'resnet50'
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.dropout_rate = 0.2
        self.batch_size = 32
        self.early_stop_patience = 5
        self.use_age = True
        self.use_sex = True
        self.use_signal_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_pretrained = False
        self.num_models = 2
        self.warmup_epochs = 10
        self.gmm_components = 2
        self.mixup_alpha = 0.5
        self.temperature = 0.5
        self.lambda_u = 50

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
        print(f"WarmUp Epochs: {self.warmup_epochs}")
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
    records = [os.path.join(data_folder, f) for f in find_records(data_folder)]
    dataset = ECGDataset(records)
    
    device = config.device
    criterion = FocalLoss(alpha=0.8, logits=True)
    scaler = torch.amp.GradScaler('cuda')
    
    # Warm-up phase: use 80% of the data for training
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    train_size = int(0.8 * num_samples)
    warmup_train_indices = indices[:train_size]
    warmup_train_subset = Subset(dataset, warmup_train_indices)
    
    models = [resnet50(pretrained=config.use_pretrained).to(device) for _ in range(config.num_models)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=config.learning_rate) for m in models]
    
    def make_weights_for_balanced_classes(subset):
        targets = [subset[i][1] for i in range(len(subset))]
        weights = np.zeros_like(targets, dtype=np.float32)
        weights[np.isclose(targets, 0.0)] = 1.0
        weights[np.isclose(targets, 1.0)] = 10.0
        return weights

    if verbose:
        print("Starting WarmUp Training...")
    for epoch in range(config.warmup_epochs):
        start_time = time.time()
        train_weights = make_weights_for_balanced_classes(warmup_train_subset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        for model_idx in range(config.num_models):
            model = models[model_idx]
            model.train()
            for (signal, meta), labels in DataLoader(warmup_train_subset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4):
                signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
                optimizers[model_idx].zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(signal, meta)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizers[model_idx])
                scaler.update()
        end_time = time.time()
        print(f"Epoch {epoch+1}/{config.warmup_epochs}, Time: {end_time - start_time:.2f}s, Loss: {loss.item():.4f}")

    # Cross-validation phase: use 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    gmms = [GaussianMixture(n_components=config.gmm_components) for _ in range(config.num_models)]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.warmup_epochs, config.num_epochs):
            start_time = time.time()
            losses = [[] for _ in range(config.num_models)]
            with torch.no_grad():
                for (signal, meta), labels in DataLoader(train_subset, batch_size=config.batch_size, num_workers=4):
                    signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
                    for i, model in enumerate(models):
                        with torch.amp.autocast('cuda'):
                            outputs = model(signal, meta)
                            loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
                        losses[i].extend(loss.cpu().numpy())
            
            labeled_indices = []
            for i, gmm in enumerate(gmms):
                gmm.fit(np.array(losses[i]).reshape(-1, 1))
                prob = gmm.predict_proba(np.array(losses[i]).reshape(-1, 1))
                prob = prob[:, gmm.means_.argmin()]
                labeled_indices.append(np.where(prob > 0.5)[0])
            
            for model_idx in range(config.num_models):
                other_model = models[1 - model_idx]
                labeled_dataset = Subset(train_subset, labeled_indices[model_idx])
                unlabeled_dataset = Subset(train_subset, np.setdiff1d(range(len(train_subset)), labeled_indices[model_idx]))
                
                model = models[model_idx]
                optimizer = optimizers[model_idx]
                
                model.train()
                for (signal_l, meta_l), labels_l in DataLoader(labeled_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4):
                    with torch.no_grad():
                        outputs = other_model(signal_l.to(device), meta_l.to(device))
                        refined_labels = torch.sigmoid(outputs).cpu()
                    
                    mixed_signal, mixed_meta, mixed_labels = signal_l, meta_l, refined_labels
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = model(mixed_signal.to(device), mixed_meta.to(device))
                        loss = criterion(outputs, mixed_labels.to(device))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            end_time = time.time()
            print(f"Epoch {epoch+1}/{config.num_epochs}, Time: {end_time - start_time:.2f}s, Loss: {loss.item():.4f}")
            print(f"Labeled samples: {len(labeled_indices[model_idx])}, Unlabeled samples: {len(unlabeled_dataset)}")
            
            # Calculate training metrics
            y_true_train = []
            y_pred_train = []
            y_prob_train = []
            with torch.no_grad():
                for (signal, meta), labels in DataLoader(train_subset, batch_size=config.batch_size, num_workers=4):
                    signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = models[0](signal, meta)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                    y_true_train.extend(labels.cpu().numpy())
                    y_pred_train.extend(preds.cpu().numpy())
                    y_prob_train.extend(probs.cpu().numpy())
            
            auroc_train = roc_auc_score(y_true_train, y_prob_train)
            auprc_train = average_precision_score(y_true_train, y_prob_train)
            f1_train = f1_score(y_true_train, y_pred_train)
            accuracy_train = accuracy_score(y_true_train, y_pred_train)
            print(f"Train AUROC: {auroc_train:.4f}, AUPRC: {auprc_train:.4f}, F1: {f1_train:.4f}, Accuracy: {accuracy_train:.4f}")
            
            # Calculate validation metrics
            y_true_val = []
            y_pred_val = []
            y_prob_val = []
            val_loss = 0.0
            with torch.no_grad():
                for (signal, meta), labels in DataLoader(val_subset, batch_size=config.batch_size, num_workers=4):
                    signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = models[0](signal, meta)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                        val_loss += criterion(outputs, labels).item()
                    y_true_val.extend(labels.cpu().numpy())
                    y_pred_val.extend(preds.cpu().numpy())
                    y_prob_val.extend(probs.cpu().numpy())
            
            val_loss /= len(val_subset)
            auroc_val = roc_auc_score(y_true_val, y_prob_val)
            auprc_val = average_precision_score(y_true_val, y_prob_val)
            f1_val = f1_score(y_true_val, y_pred_val)
            accuracy_val = accuracy_score(y_true_val, y_pred_val)
            print(f"Validation Loss: {val_loss:.4f}, AUROC: {auroc_val:.4f}, AUPRC: {auprc_val:.4f}, F1: {f1_val:.4f}, Accuracy: {accuracy_val:.4f}")
        
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(model_folder, models)
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    print("Early stopping triggered")
                    break

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # Initialize single model for inference
    model = resnet50().to(config.device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pth'), weights_only=True))
    return model

# Run your trained model.
def run_model(record, model, verbose):
    # Modified to work with single model
    model.eval()
    
    # Extract features
    signal, meta_features = extract_features(record)
    
    # Prepare inputs
    signal = torch.from_numpy(np.expand_dims(signal, 0)).to(config.device)
    meta_features = torch.from_numpy(np.expand_dims(meta_features, 0)).to(config.device)
    
    # Get prediction
    with torch.no_grad():
        output = model(signal, meta_features)
        probability = torch.sigmoid(output).item()
    
    return probability > 0.5, probability

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
def save_model(model_folder, models):
    os.makedirs(model_folder, exist_ok=True)
    # Save only the first model while maintaining dual-model training
    torch.save(models[0].state_dict(), os.path.join(model_folder, 'model.pth'))

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
