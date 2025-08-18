import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from helper_code import *

class ECGDataset(Dataset):
    GLOBAL_AGE_MEAN = 54.6773
    GLOBAL_AGE_STD = 19.9747

    def __init__(self, dataset_name, data_folder='./external_data', is_training=True, config=None):
        self.is_training = is_training
        self.config = config
        self.dataset_name = dataset_name
        self.data_folder = data_folder

        self.dataset_to_domain_label = {
            'CODE15': 0,
            'CSPC': 1,
            'CSPC_extra': 2,
            'Chapman_Shaoxing': 3,
            'Georgia': 4,
            'Ningbo': 5,
            'PTB': 6,
            'ST_Petersburg': 7,
            'PTBXL': 8,
            'SaMiTrop': 9
        }

        self.file_path = os.path.join(self.data_folder, f"{self.dataset_name}_data.hdf5")

        if self.dataset_name not in self.dataset_to_domain_label:
            print(f"Warning: Unknown dataset_name: {self.dataset_name}. Domain label will be handled as -1.")

        try:
            self.hdf5_file = h5py.File(self.file_path, 'r')
            self.num_records = len(self.hdf5_file['signals'])
        except Exception as e:
            print(f"Error opening HDF5 file at {self.file_path}: {str(e)}")
            raise

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        signal = self.hdf5_file['signals'][idx] # Read signal data for this record
        signal = signal.T
        meta_features = self.hdf5_file['meta_features'][idx] # Read meta features for this record
        label = self.hdf5_file['labels'][idx] # Read label for this record
        domain_label = self.hdf5_file['domain_labels'][idx] # Read domain label for this record
        
        # if len(meta_features) > 0:
        #     meta_features[0] = (meta_features[0] - self.GLOBAL_AGE_MEAN) / self.GLOBAL_AGE_STD

        if self.is_training:            
            if self.config.augmentation['noise']['use'] and np.random.rand() < self.config.augmentation['noise']['prob']:
                signal = self._add_noise(signal)
                
            if self.config.augmentation['scaling']['use'] and np.random.rand() < self.config.augmentation['scaling']['prob']:
                signal = self._scaling(signal)
                
            if self.config.augmentation['flip']['use'] and np.random.rand() < self.config.augmentation['flip']['prob']:
                signal = np.ascontiguousarray(self._flip(signal))
                
            if self.config.augmentation['shift']['use'] and np.random.rand() < self.config.augmentation['shift']['prob']:
                signal = self._shift(signal)
                
            if self.config.augmentation['drop']['use'] and np.random.rand() < self.config.augmentation['drop']['prob']:
                signal = self._drop(signal)
            
            if self.config.augmentation['power_noise']['use'] and np.random.rand() < self.config.augmentation['power_noise']['prob']:
                signal = self._add_power_noise(signal)
                
            if self.config.augmentation['cutout']['use'] and np.random.rand() < self.config.augmentation['cutout']['prob']:
                signal = self._cutout(signal)
                
            if self.config.augmentation['time_warp']['use'] and np.random.rand() < self.config.augmentation['time_warp']['prob']:
                signal = self._time_wrapping(signal)
                
            if self.config.augmentation['lead_mixing']['use'] and np.random.rand() < self.config.augmentation['lead_mixing']['prob']:
                signal = self._lead_mixing_augmentation(signal, self.config.augmentation['lead_mixing']['lambda'])
                
            if self.config.augmentation['baseline_wander']['use'] and np.random.rand() < self.config.augmentation['baseline_wander']['prob']:
                signal = self._baseline_wander(signal)

            signal = np.ascontiguousarray(signal).astype(np.float32)
            meta_features = meta_features.astype(np.float32)
            # domain_label = np.array(domain_label) # Convert to numpy array
            domain_label = domain_label.astype(np.float32) # Ensure domain_label is float32
            
        features = [signal, meta_features]

        return features, label, domain_label, idx
        
    def close(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()
            print("HDF5 file closed.")

    def __del__(self):
        self.close()

    def _add_noise(self, signal):
        noise = np.random.normal(0, self.config.augmentation['noise']['std'], signal.shape)
        signal += noise
        return signal
        
    def _scaling(self, signal):
        scaling_factors = np.random.uniform(self.config.augmentation['scaling']['min'], self.config.augmentation['scaling']['max'], signal.shape[0])
        signal *= scaling_factors[:, np.newaxis]
        signal = np.clip(signal, -1e4, 1e4)
        return signal
        
    def _flip(self, signal):
        signal *= -1
        return signal
        
    def _shift(self, signal):
        length = signal.shape[1]
        max_shift = int(length * self.config.augmentation['shift']['max_ratio'])
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        shifted = np.roll(signal, shift_amount, axis=1)
        return shifted
        
    def _drop(self, signal):
        mask = np.random.rand(*signal.shape) > self.config.augmentation['drop']['max_prob']
        signal *= mask
        return signal
        
    def _cutout(self, signal):
        length = signal.shape[1]
        max_cutout = int(length * self.config.augmentation['cutout']['max_ratio'])
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
            amplitude = min(self.config.augmentation['power_noise']['amplitude'] * signal_std, 0.1)
            length = signal.shape[1]
            t = np.arange(length) / 500.0
            phase = np.random.uniform(0, 2 * np.pi)
            power_noise = amplitude * np.sin(2 * np.pi * 50 * t + phase)
            signal += power_noise
            signal = np.clip(signal, -1e4, 1e4)

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

            return signal
            
        except Exception as e:
            print(f"Lead mixing failed: {str(e)}")
            return signal
            
    def _baseline_wander(self, signal):
        t = np.arange(signal.shape[1])
        freq = np.random.uniform(self.config.augmentation['baseline_wander']['min_freq'], self.config.augmentation['baseline_wander']['max_freq'])
        amp = self.config.augmentation['baseline_wander']['amp_ratio'] * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t)
        signal += drift

        return signal

    def _time_wrapping(self, signal):
        original_length = signal.shape[1]
        original_freq = 500
        new_freq = np.random.randint(self.config.augmentation['time_warp']['min_hz'], self.config.augmentation['time_warp']['max_hz'] + 1)
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
