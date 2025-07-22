import os
import numpy as np
import torch
import psutil
from helper_code import *
from scipy.signal import butter, filtfilt, resample
from memory_profiler import profile

def print_memory_usage(extra_info=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    print(f"\n====== Memory Usage {extra_info} ======")
    print(f"Process RSS (Resident Set Size, actual physical memory used by process): {mem_info.rss / 1024 ** 3:.2f} GB")
    print(f"Process VMS (Virtual Memory Size, total virtual memory used by process): {mem_info.vms / 1024 ** 3:.2f} GB")
    print(f"System Used (Total RAM used by the system): {vm.used / 1024 ** 3:.2f} GB")
    print(f"System Available (RAM available for new processes): {vm.available / 1024 ** 3:.2f} GB / {vm.total / 1024 ** 3:.2f} GB (Total System RAM)")
    print(f"Memory Used % (Percentage of System RAM used): {vm.percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated (GPU memory currently allocated): {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU Memory Cached (GPU memory currently cached): {torch.cuda.memory_cached() / 1024 ** 3:.2f} GB")
    print("==========================================\n")

def print_model_parameters(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def initialize_filters():
    target_fs = 500
    nyquist = 0.5 * target_fs

    highpass_cutoff = 1 / nyquist
    b_high, a_high = butter(2, highpass_cutoff, btype='high')

    lowpass_cutoff = 30 / nyquist
    b_low, a_low = butter(2, lowpass_cutoff, btype='low')

    notch_freq_50 = 50
    bandwidth_50 = 5
    freq_50 = notch_freq_50 / nyquist
    bw_50 = bandwidth_50 / nyquist
    b_notch50, a_notch50 = butter(2, [freq_50 - bw_50/2, freq_50 + bw_50/2], btype='bandstop')

    notch_freq_60 = 60
    bandwidth_60 = 5
    freq_60 = notch_freq_60 / nyquist
    bw_60 = bandwidth_60 / nyquist
    b_notch60, a_notch60 = butter(2, [freq_60 - bw_60/2, freq_60 + bw_60/2], btype='bandstop')

    return (b_high, a_high), (b_low, a_low), (b_notch50, a_notch50), (b_notch60, a_notch60)

@profile 
def data_preprocess(record, config=None, highpass_filter_params=None, lowpass_filter_params=None, notch50_filter_params=None, notch60_filter_params=None):
    if config is None:
        config = globals().get('config')
    os.makedirs(config.cache_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(record))[0]
    
    signal_path = os.path.join(config.cache_folder, f'{base_name}_signal.npy')
    if os.path.exists(signal_path):
        return

    header = load_header(record)
    signal, fields = load_signals(record)
    channels = fields['sig_name']
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = reorder_signal(signal, channels, reference_channels)
    signal = signal.astype(np.float32, copy=False)
    
    # if np.isnan(signal).any() or np.isinf(signal).any():
    #     print("WARNING: Raw signal contains NaN/Inf values")

    original_fs = get_sampling_frequency(header)
    target_fs = 500
    if original_fs != target_fs:
        target_length = int(signal.shape[0] * target_fs / original_fs)
        try:
            signal = resample(signal, target_length, axis=0).astype(np.float32, copy=False)
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
        
    if highpass_filter_params is not None:
        b_high, a_high = highpass_filter_params
        signal = filtfilt(b_high, a_high, signal, axis=0)
    
    if lowpass_filter_params is not None:
        b_low, a_low = lowpass_filter_params
        signal = filtfilt(b_low, a_low, signal, axis=0)
    
    # if notch50_filter_params is not None:
    #     b_notch50, a_notch50 = notch50_filter_params
    #     signal = filtfilt(b_notch50, a_notch50, signal, axis=0)
    
    # if notch60_filter_params is not None:
    #     b_notch60, a_notch60 = notch60_filter_params
    #     signal = filtfilt(b_notch60, a_notch60, signal, axis=0)
        
    if np.isnan(signal).any():
        print("WARNING: Signal contains NaN values")

    signal = np.ascontiguousarray(signal.T)

    signal_mean = np.mean(signal, axis=0)
    signal_std = np.std(signal, axis=0)
    signal_std[signal_std < 1e-8] = 1.0
    signal -= signal_mean
    signal /= signal_std

    np.save(signal_path, signal.astype(np.float32, copy=False))

def delete_record_files(record, config=None):
    if config is None:
        config = globals().get('config')
    base_name = os.path.splitext(os.path.basename(record))[0]
    signal_path = os.path.join(config.cache_folder, f'{base_name}_signal.npy')

    if os.path.exists(signal_path):
        os.remove(signal_path)

def save_model(model_folder, state_dict, config=None, fold=None):
    if config is None:
        config = globals().get('config')
    os.makedirs(model_folder, exist_ok=True)
    
    config_dict = config.__dict__.copy()
    config_dict['meta_input_dim'] = config.get_meta_feature_dim()
    checkpoint = {
        'state_dict': state_dict,
        'config': config_dict
    }
    
    if fold is not None:
        filename = os.path.join(model_folder, f'model_fold{fold}.pth')
    else:
        filename = os.path.join(model_folder, 'model.pth')
        
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")
