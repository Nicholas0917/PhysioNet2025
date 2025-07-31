import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from helper_code import *

class ECGDataset(Dataset):
    def __init__(self, hdf5_path, is_training=True, config=None):
        self.hdf5_path = hdf5_path
        self.is_training = is_training
        self.config = config
        
        try:
            self.hdf5_file = h5py.File(hdf5_path, 'r')
            self.num_records = len(self.hdf5_file['signals'])
        except Exception as e:
            print(f"Error opening HDF5 file at {hdf5_path}: {str(e)}")
            raise

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        signal = self.hdf5_file['signals'][idx] # Read signal data for this record
        signal = signal.T
        meta_features = self.hdf5_file['meta_features'][idx] # Read meta features for this record
        label = self.hdf5_file['labels'][idx] # Read label for this record
            
        # if np.isnan(signal).any() or np.isinf(signal).any():
        #     print(f"WARNING: Signal contains NaN/Inf values in record {idx}")
        #     print(f"Signal stats - min: {np.nanmin(signal):.4f}, max: {np.nanmax(signal):.4f}, mean: {np.nanmean(signal):.4f}")
        # if np.isnan(meta_features).any() or np.isinf(meta_features).any():
        #     print(f"WARNING: Meta features contain NaN/Inf values in record {idx}")
        # if np.isnan(label) or np.isinf(label):
        #     print(f"WARNING: Label contains NaN/Inf in record {record}")
        
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
        
    def close(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()
            print("HDF5 file closed.")

    def __del__(self):
        self.close()

    def _add_noise(self, signal):
        noise = np.random.normal(0, self.config.noise_std, signal.shape)
        signal += noise
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _add_noise")
        return signal
        
    def _scaling(self, signal):
        scaling_factors = np.random.uniform(self.config.scaling_min, self.config.scaling_max, signal.shape[0])
        signal *= scaling_factors[:, np.newaxis]
        signal = np.clip(signal, -1e4, 1e4)
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _scaling")
        return signal
        
    def _flip(self, signal):
        signal *= -1
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _flip")
        return signal
        
    def _shift(self, signal):
        length = signal.shape[1]
        max_shift = int(length * self.config.shift_max_ratio)
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        shifted = np.roll(signal, shift_amount, axis=1)
        # if np.isnan(shifted).any():
        #     print("WARNING: NaN detected after _shift")
        return shifted
        
    def _drop(self, signal):
        mask = np.random.rand(*signal.shape) > self.config.drop_max_prob
        signal *= mask
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _drop")
        return signal
        
    def _sine_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.sine_min_freq, self.config.sine_max_freq)
        amp = np.random.uniform(0, self.config.sine_max_amp)
        sine = amp * np.sin(2 * np.pi * freq * t)
        signal += sine[np.newaxis, :]
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _sine_wave")
        return signal
        
    def _square_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.square_min_freq, self.config.square_max_freq)
        amp = np.random.uniform(0, self.config.square_max_amp)
        square = amp * np.sign(np.sin(2 * np.pi * freq * t))
        signal += square[np.newaxis, :]
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _square_wave")
        return signal
        
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
            
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _cutout")
        return signal
        
    def _add_power_noise(self, signal):
        signal_std = np.std(signal)
        if signal_std > 0:
            amplitude = min(self.config.power_noise_amplitude * signal_std, 0.1)
            length = signal.shape[1]
            t = np.arange(length) / 500.0
            phase = np.random.uniform(0, 2 * np.pi)
            power_noise = amplitude * np.sin(2 * np.pi * 50 * t + phase)
            signal += power_noise
            signal = np.clip(signal, -1e4, 1e4)
            
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _add_power_noise")
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
            
            # Perform in-place update for the final augmented signal
            signal = (1 - lambda_val) * signal + lambda_val * new_signal
            signal = np.clip(signal, -1e4, 1e4)
            
            # if np.isnan(signal).any():
            #     print("WARNING: NaN detected after _lead_mixing_augmentation")
            return signal
            
        except Exception as e:
            print(f"Lead mixing failed: {str(e)}")
            return signal
            
    def _baseline_wander(self, signal):
        t = np.arange(signal.shape[1])
        freq = np.random.uniform(self.config.baseline_wander_min_freq, self.config.baseline_wander_max_freq)
        amp = self.config.baseline_wander_amp_ratio * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t)
        signal += drift
        # if np.isnan(signal).any():
        #     print("WARNING: NaN detected after _baseline_wander")
        return signal

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
        
        # if np.isnan(result).any() or np.isinf(result).any():
        #     print("WARNING: NaN/Inf detected after _time_wrapping")
        return result

class ExternalDataset(Dataset):
    def __init__(self, dataset_name, data_folder='./external_data', is_training=True, config=None):
        self.dataset_name = dataset_name
        self.data_folder = data_folder
        self.file_path = os.path.join(self.data_folder, f"{dataset_name}_data.hdf5")
        self.is_training = is_training
        self.config = config
        
        self.dataset_to_label = {
            'CODE15': 0,
            'CSPC': 1,
            'CSPC_extra': 2,
            'Chapman_Shaoxing': 3,
            'Georgia': 4,
            'Ningbo': 5,
            'PTB': 6,
            'ST_Petersburg': 7
        }
        
        if self.dataset_name not in self.dataset_to_label:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}. Must be one of {list(self.dataset_to_label.keys())}")

        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f['signals'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            signal = f['signals'][idx]
            signal = signal.T
            meta_features_data = f['meta_features'][idx] # Directly read meta_features for the given index

        # Check for NaN in age (assuming age is the first element)
        if np.isnan(meta_features_data[0]):
            print(f"WARNING: NaN detected in age for dataset: {self.dataset_name}, sample index: {idx}. Replacing with 0.")
            meta_features_data[0] = 0.0 # Replace NaN with 0

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

        # meta_features_data should already be a numpy array with age and sex
        # Assuming meta_features_data is a 1D array like [age, sex_one_hot_encoded]
        # If meta_features_data is just [age, sex_value], then adjust accordingly.
        # Based on the original code, meta_features is an array of [age, sex]
        meta_features = meta_features_data.astype(np.float32)
        
        # Check for NaN in signal before returning
        if np.isnan(signal).any():
            raise RuntimeError(f"NaN detected in signal for ExternalDataset, dataset: {self.dataset_name}, sample index: {idx}. Skipping this sample.")

        label = self.dataset_to_label[self.dataset_name]

        features = [signal, meta_features]
        return features, label
        
    def _add_noise(self, signal):
        noise = np.random.normal(0, self.config.noise_std, signal.shape)
        signal += noise
        return signal
        
    def _scaling(self, signal):
        scaling_factors = np.random.uniform(self.config.scaling_min, self.config.scaling_max, signal.shape[0])
        signal *= scaling_factors[:, np.newaxis]
        signal = np.clip(signal, -1e4, 1e4)
        return signal
        
    def _flip(self, signal):
        signal *= -1
        return signal
        
    def _shift(self, signal):
        length = signal.shape[1]
        max_shift = int(length * self.config.shift_max_ratio)
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        shifted = np.roll(signal, shift_amount, axis=1)
        return shifted
        
    def _drop(self, signal):
        mask = np.random.rand(*signal.shape) > self.config.drop_max_prob
        signal *= mask
        return signal
        
    def _sine_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.sine_min_freq, self.config.sine_max_freq)
        amp = np.random.uniform(0, self.config.sine_max_amp)
        sine = amp * np.sin(2 * np.pi * freq * t)
        signal += sine[np.newaxis, :]
        return signal
        
    def _square_wave(self, signal):
        length = signal.shape[1]
        t = np.arange(length)
        freq = np.random.uniform(self.config.square_min_freq, self.config.square_max_freq)
        amp = np.random.uniform(0, self.config.square_max_amp)
        square = amp * np.sign(np.sin(2 * np.pi * freq * t))
        signal += square[np.newaxis, :]
        return signal
        
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
        return signal
        
    def _add_power_noise(self, signal):
        signal_std = np.std(signal)
        if signal_std > 0:
            amplitude = min(self.config.power_noise_amplitude * signal_std, 0.1)
            length = signal.shape[1]
            t = np.arange(length) / 500.0
            phase = np.random.uniform(0, 2 * np.pi)
            power_noise = amplitude * np.sin(2 * np.pi * 50 * t + phase)
            signal += power_noise
            signal = np.clip(signal, -1e4, 1e4)
        return signal

    def _lead_mixing_augmentation(self, signal, lambda_val=0.2):
        try:
            # Step 1: Check if the original signal contains NaN or Inf
            if not np.isfinite(signal).all():
                print("[Debug] Input signal contains NaN or Inf!")

            # Step 2: Calculate mean and standard deviation
            signal_std = np.std(signal, axis=1, keepdims=True)
            signal_mean = np.mean(signal, axis=1, keepdims=True)

            # Step 3: Check if any standard deviation is 0
            if np.any(signal_std < 1e-8):
                print(f"[Debug] {np.sum(signal_std < 1e-8)} leads have very small std (approx. 0)")

            # Step 4: Standardize and replace std with a safe minimum
            safe_std = np.where(signal_std < 1e-8, 1e-8, signal_std)
            normalized = (signal - signal_mean) / safe_std

            # Step 5: Recheck if NaN or Inf appeared in normalized signal
            if not np.isfinite(normalized).all():
                print("[Debug] NaN or Inf appeared in normalized signal")

            # Step 6: Find leads with valid standard deviation
            std_after_norm = np.std(normalized, axis=1)
            non_constant_leads_mask = std_after_norm > 1e-8
            if not np.any(non_constant_leads_mask):
                print("[Debug] All leads have std close to 0, skipping augmentation")
                return signal

            # Step 7: Calculate correlation coefficient matrix
            normalized_for_corr = normalized[non_constant_leads_mask]
            original_indices_of_non_constant_leads = np.where(non_constant_leads_mask)[0]
            corr_matrix = np.corrcoef(normalized_for_corr)

            # Step 8: Check if corr_matrix contains NaN
            if not np.isfinite(corr_matrix).all():
                print("[Debug] NaN exists in corr_matrix, replacing")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Step 9: Build weighted matrix A
            A = np.abs(corr_matrix)
            np.fill_diagonal(A, 0)

            row_sums = A.sum(axis=1, keepdims=True)
            if np.any(row_sums < 1e-8):
                print(f"[Debug] {np.sum(row_sums < 1e-8)} rows have very small sum")

            safe_row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
            A = A / safe_row_sums

            # Step 10: Generate new signal
            new_signal = np.copy(signal)
            for i_idx, i_orig in enumerate(original_indices_of_non_constant_leads):
                weights_in_A = A[i_idx, :]
                full_weights = np.zeros(signal.shape[0])
                for j_idx, j_orig in enumerate(original_indices_of_non_constant_leads):
                    full_weights[j_orig] = weights_in_A[j_idx]

                if np.any(full_weights > 0):
                    new_signal[i_orig] = np.sum(signal * full_weights.reshape(-1, 1), axis=0)

            # Step 11: Mix original and new signal
            signal = (1 - lambda_val) * signal + lambda_val * new_signal
            signal = np.clip(signal, -1e4, 1e4)

            # Step 12: Final check for anomalies in output
            if not np.isfinite(signal).all():
                print("[Debug] Output signal contains NaN or Inf")

            return signal

        except Exception as e:
            print(f"[Error] Lead mixing failed: {str(e)}")
            return signal

            
    def _baseline_wander(self, signal):
        t = np.arange(signal.shape[1])
        freq = np.random.uniform(self.config.baseline_wander_min_freq, self.config.baseline_wander_max_freq)
        amp = self.config.baseline_wander_amp_ratio * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t)
        signal += drift
        return signal

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
        return result
