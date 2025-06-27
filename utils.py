import os
import numpy as np
import torch
import psutil
from helper_code import *
from scipy.signal import butter, filtfilt, resample

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

def data_preprocess(record, config=None):
    if config is None:
        config = globals().get('config')
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
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = reorder_signal(signal, channels, reference_channels)
    signal = signal.astype(np.float32)
    
    if np.isnan(signal).any() or np.isinf(signal).any():
        print("WARNING: Raw signal contains NaN/Inf values")

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
        
    nyquist = 0.5 * 500
    highpass_cutoff = 1 / nyquist
    b, a = butter(2, highpass_cutoff, btype='high')
    signal = filtfilt(b, a, signal, axis=0)
    
    lowpass_cutoff = 30 / nyquist
    b, a = butter(2, lowpass_cutoff, btype='low')
    signal = filtfilt(b, a, signal, axis=0)
    
    notch_freq = 50
    bandwidth = 5
    freq = notch_freq / nyquist
    bw = bandwidth / nyquist
    b, a = butter(2, [freq - bw/2, freq + bw/2], btype='bandstop')
    signal = filtfilt(b, a, signal, axis=0)
    
    notch_freq = 60
    bandwidth = 5
    freq = notch_freq / nyquist
    bw = bandwidth / nyquist
    b, a = butter(2, [freq - bw/2, freq + bw/2], btype='bandstop')
    signal = filtfilt(b, a, signal, axis=0)
        
    if np.isnan(signal).any():
        print("WARNING: Signal contains NaN values")

    signal = np.ascontiguousarray(signal.T)
    signal_mean = np.mean(signal, axis=0)
    signal_std = np.std(signal, axis=0)
    signal_std[signal_std < 1e-8] = 1.0
    signal = (signal - signal_mean) / signal_std
    
    if np.isnan(signal).any() or np.isinf(signal).any():
        print("WARNING: Signal contains NaN/Inf after normalization")
        signal = np.nan_to_num(signal, nan=0.0, posinf=1e4, neginf=-1e4)

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

    np.save(signal_path, signal.astype(np.float32))

def delete_record_files(record, config=None):
    if config is None:
        config = globals().get('config')
    base_name = os.path.splitext(os.path.basename(record))[0]
    signal_path = os.path.join(config.cache_folder, f'{base_name}_signal.npy')

    if os.path.exists(signal_path):
        os.remove(signal_path)

def save_model(model_folder, state_dict, config=None):
    if config is None:
        config = globals().get('config')
    model_dir = os.path.join(model_folder, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    config_dict = config.__dict__.copy()
    config_dict['meta_input_dim'] = config.get_meta_feature_dim()
    checkpoint = {
        'state_dict': state_dict,
        'config': config_dict
    }
    filename = os.path.join(model_dir, 'model.pth')
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")
