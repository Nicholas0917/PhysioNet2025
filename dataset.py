import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from helper_code import *

class ECGDataset(Dataset):
    def __init__(self, records, is_training=True, config=None):
        self.records = records
        self.is_training = is_training
        self.config = config
        self.cache_folder = config.cache_folder if config else './tmp'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        base_name = os.path.splitext(os.path.basename(self.records[idx]))[0]
        signal_path = os.path.join(self.cache_folder, f"{base_name}_signal.npy")
        
        try:
            with open(signal_path, 'rb') as f:
                signal = np.load(f)
                if np.isnan(signal).any() or np.isinf(signal).any():
                    print(f"WARNING: Signal contains NaN/Inf values in {signal_path}")
                    print(f"Signal stats - min: {np.nanmin(signal):.4f}, max: {np.nanmax(signal):.4f}, mean: {np.nanmean(signal):.4f}")
        except Exception as e:
            print(f"Error loading {signal_path}: {str(e)}")
            raise
        
        record = self.records[idx]
        label = float(load_label(record))
        if np.isnan(label) or np.isinf(label):
            print(f"WARNING: Label contains NaN/Inf in record {record}")
        
        header = load_header(record)
        age = get_age(header) if self.config.use_age else 0
        sex = get_sex(header) if self.config.use_sex else 'Unknown'
        
        one_hot_encoding_sex = np.zeros(3, dtype=np.bool_)
        if sex == 'Female':
            one_hot_encoding_sex[0] = True
        elif sex == 'Male':
            one_hot_encoding_sex[1] = True
        else:
            one_hot_encoding_sex[2] = True

        meta_features = np.empty(self.config.get_meta_feature_dim(), dtype=np.float32)
        ptr = 0
        
        if self.config.use_age:
            meta_features[ptr] = age
            ptr += 1
        if self.config.use_sex:
            meta_features[ptr:ptr+3] = one_hot_encoding_sex
            ptr += 3
        if self.config.use_signal_stats:
            valid_samples = np.isfinite(signal).sum()
            meta_features[ptr] = np.nanmean(signal) if valid_samples > 0 else 0.0
            meta_features[ptr+1] = np.nanstd(signal) if valid_samples > 1 else 0.0
            ptr += 2

        if self.is_training:
            if self.config.use_noise_aug and np.random.rand() < self.config.noise_aug_prob:
                signal = self._add_noise(signal)
                
            if self.config.use_scaling_aug and np.random.rand() < self.config.scaling_aug_prob:
                signal = self._scaling(signal)
                
            if self.config.use_flip_aug and np.random.rand() < self.config.flip_aug_prob:
                signal = np.ascontiguousarray(self._flip(signal))
                
            if self.config.use_shift_aug and np.random.rand() < self.config.shift_aug_prob:
                signal = self._shift(signal)
                
            if self.config.use_drop_aug and np.random.rand() < self.config.drop_aug_prob:
                signal = self._drop(signal)
            
            if self.config.add_power_noise and np.random.rand() < self.config.power_noise_prob:
                signal = self._add_power_noise(signal)
                
            if self.config.use_sine_wave_aug and np.random.rand() < self.config.sine_aug_prob:
                signal = self._sine_wave(signal)
                
            if self.config.use_square_wave_aug and np.random.rand() < self.config.square_aug_prob:
                signal = self._square_wave(signal)
                
            if self.config.use_cutout_aug and np.random.rand() < self.config.cutout_aug_prob:
                signal = self._cutout(signal)
                
            if self.config.use_time_warp_aug and np.random.rand() < self.config.time_wrap_prob:
                signal = self._time_wrapping(signal)
                
            if self.config.use_lead_mixing_aug and np.random.rand() < self.config.lead_mixing_prob:
                signal = self._lead_mixing_augmentation(signal, self.config.lead_mixing_lambda)
                
            if self.config.use_baseline_wander and np.random.rand() < self.config.baseline_wander_prob:
                signal = self._baseline_wander(signal)

            signal = np.ascontiguousarray(signal).astype(np.float32)
            meta_features = meta_features.astype(np.float32)
            
        features = [signal, meta_features]

        return features, label
        
    def _add_noise(self, signal):
        noise = np.random.normal(0, self.config.noise_std, signal.shape)
        augmented = signal + noise
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _add_noise")
        return augmented
        
    def _scaling(self, signal):
        scaling_factors = np.random.uniform(self.config.scaling_min, self.config.scaling_max, signal.shape[0])
        scaled = signal * scaling_factors[:, np.newaxis]
        scaled = np.clip(scaled, -1e4, 1e4)
        if np.isnan(scaled).any():
            print("WARNING: NaN detected after _scaling")
        return scaled
        
    def _flip(self, signal):
        flipped = signal * -1
        if np.isnan(flipped).any():
            print("WARNING: NaN detected after _flip")
        return flipped
        
    def _shift(self, signal):
        length = signal.shape[1]
        max_shift = int(length * self.config.shift_max_ratio)
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        shifted = np.roll(signal, shift_amount, axis=1)
        if np.isnan(shifted).any():
            print("WARNING: NaN detected after _shift")
        return shifted
        
    def _drop(self, signal):
        mask = np.random.rand(*signal.shape) > self.config.drop_max_prob
        dropped = signal * mask
        if np.isnan(dropped).any():
            print("WARNING: NaN detected after _drop")
        return dropped
        
    def _sine_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.sine_min_freq, self.config.sine_max_freq)
        amp = np.random.uniform(0, self.config.sine_max_amp)
        sine = amp * np.sin(2 * np.pi * freq * t)
        augmented = signal + sine[np.newaxis, :]
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _sine_wave")
        return augmented
        
    def _square_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.square_min_freq, self.config.square_max_freq)
        amp = np.random.uniform(0, self.config.square_max_amp)
        square = amp * np.sign(np.sin(2 * np.pi * freq * t))
        augmented = signal + square[np.newaxis, :]
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _square_wave")
        return augmented
        
    def _cutout(self, signal):
        length = signal.shape[1]
        max_cutout = int(length * self.config.cutout_max_ratio)
        if max_cutout == 0:
            return signal
            
        num_leads = signal.shape[0]
        num_cutout_leads = np.random.randint(1, num_leads + 1)
        cutout_leads = np.random.choice(num_leads, num_cutout_leads, replace=False)
        
        for lead in cutout_leads:
            cutout_width = np.random.randint(1, max_cutout + 1)
            start = np.random.randint(0, length - cutout_width + 1)
            signal[lead, start:start+cutout_width] = 0
            
        if np.isnan(signal).any():
            print("WARNING: NaN detected after _cutout")
        return signal
        
    def _add_power_noise(self, signal):
        signal_std = np.std(signal)
        if signal_std > 0:
            amplitude = min(self.config.power_noise_amplitude * signal_std, 0.1)
            length = signal.shape[1]
            t = np.arange(length) / 500.0
            phase = np.random.uniform(0, 2 * np.pi)
            power_noise = amplitude * np.sin(2 * np.pi * 50 * t + phase)
            signal = signal + power_noise
            signal = np.clip(signal, -1e4, 1e4)
            
        if np.isnan(signal).any():
            print("WARNING: NaN detected after _add_power_noise")
        return signal

    def _lead_mixing_augmentation(self, signal, lambda_val=0.2):
        try:
            signal_std = np.std(signal, axis=1, keepdims=True)
            signal_mean = np.mean(signal, axis=1, keepdims=True)
            signal_std[signal_std < 1e-8] = 1e-8
            normalized = (signal - signal_mean) / signal_std
            
            corr_matrix = np.corrcoef(normalized)
            
            A = np.abs(corr_matrix)
            np.fill_diagonal(A, 0)
            
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-8] = 1.0
            A = A / row_sums
            
            new_signal = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                if np.any(A[i] > 0):
                    weights = A[i].reshape(-1, 1)
                    new_signal[i] = np.sum(signal * weights, axis=0)
            
            augmented = (1 - lambda_val) * signal + lambda_val * new_signal
            augmented = np.clip(augmented, -1e4, 1e4)
            
            if np.isnan(augmented).any():
                print("WARNING: NaN detected after _lead_mixing_augmentation")
            return augmented
            
        except Exception as e:
            print(f"Lead mixing failed: {str(e)}")
            return signal
            
    def _baseline_wander(self, signal):
        t = np.arange(signal.shape[1])
        freq = np.random.uniform(self.config.baseline_wander_min_freq, self.config.baseline_wander_max_freq)
        amp = self.config.baseline_wander_amp_ratio * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t)
        augmented = signal + drift
        if np.isnan(augmented).any():
            print("WARNING: NaN detected after _baseline_wander")
        return augmented

    def _time_wrapping(self, signal):
        original_length = signal.shape[1]
        original_freq = 500
        new_freq = np.random.randint(self.config.time_wrap_min_hz, self.config.time_wrap_max_hz + 1)
        new_length = int(original_length * new_freq / original_freq)
        
        if new_length == original_length:
            return signal.copy()
        
        old_indices = np.linspace(0, original_length - 1, new_length)
        x_old = np.arange(original_length)
        result = np.zeros((signal.shape[0], original_length))
        
        for i in range(signal.shape[0]):
            resampled = np.interp(old_indices, x_old, signal[i])
            if new_length > original_length:
                result[i] = resampled[:original_length]
            else:
                result[i, :new_length] = resampled
        
        if np.isnan(result).any() or np.isinf(result).any():
            print("WARNING: NaN/Inf detected after _time_wrapping")
        return result
