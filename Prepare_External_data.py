import wfdb
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
import h5py
import argparse
import os
import glob
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor

def contains_nan(signal):
    return np.isnan(signal).any()

def has_zero_lead(signal):
    return any(np.all(lead == 0) for lead in signal)

def is_low_variance(signal, threshold=1e-6):
    # Check variance for each lead (axis=0 for samples, axis=1 for leads)
    # Assuming signal is (samples, leads)
    return any(np.std(lead) < threshold for lead in signal.T)

def is_amplitude_normal(signal, low=0.05, high=5.0):
    max_amp = np.max(np.abs(signal))
    return low <= max_amp <= high

def has_valid_peaks(signal, lead_index=1, min_peaks=3):
    lead = signal[lead_index]
    peaks, _ = find_peaks(lead, distance=50, height=np.std(lead))
    return len(peaks) >= min_peaks

def find_wfdb_files(data_dir):
    """
    Finds all .hea files within the given directory and its subdirectories.
    Returns a list of file paths without the .hea extension.
    """
    hea_files = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
    file_paths_without_ext = [file_path[:-4] for file_path in hea_files]
    return file_paths_without_ext

# Helper function for reorder_signal
def normalize_names(input_channels, output_channels):
    normalized_output_channels = []
    for out_ch in output_channels:
        found = False
        for in_ch in input_channels:
            if out_ch.lower() == in_ch.lower():
                normalized_output_channels.append(in_ch)
                found = True
                break
        if not found:
            normalized_output_channels.append(out_ch) # Keep original if not found, or handle error as needed
    return normalized_output_channels

def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        output_channels = normalize_names(input_channels, output_channels)

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype

        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal

def get_age(record):
    age = None
    for line in record.comments:
        if 'Age:' in line:
            try:
                age = int(line.split('Age:')[1].strip())
            except ValueError:
                pass
            break
    return age

def get_sex(record):
    sex = None
    for line in record.comments:
        if 'Sex:' in line:
            sex_str = line.split('Sex:')[1].strip().upper()
            if sex_str == 'M':
                sex = 'Male'
            elif sex_str == 'F':
                sex = 'Female'
            break
    
    one_hot_encoding_sex = [False, False, False] # [Female, Male, Unknown]
    if sex == 'Female':
        one_hot_encoding_sex[0] = True
    elif sex == 'Male':
        one_hot_encoding_sex[1] = True
    else:
        one_hot_encoding_sex[2] = True
    
    return np.array(one_hot_encoding_sex, dtype=np.float32)

def process_single_file(file_path, standard_channels, fixed_domain_label):
    """
    Processes a single WFDB file, performs signal processing and quality checks,
    and returns a list of valid (signal_segment, meta_feature, domain_label, label) tuples.
    """
    processed_segments_data = []
    record_name = os.path.basename(file_path)
    
    # Use the fixed domain label passed from main
    domain_label = fixed_domain_label
    
    # Set label for chagas data (all chagas data is 0)
    label = 0 # All chagas data has label 0

    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal
        fs = record.fs
        input_channels = record.sig_name

        # Get age and sex
        age = get_age(record)
        sex = get_sex(record)

        # Reorder signal
        signal = reorder_signal(signal, input_channels, standard_channels)

        # Resample to 500 Hz
        if fs != 500:
            # Ensure signal is float before resampling to avoid issues with integer division
            signal = signal.astype(np.float32) 
            signal = resample_poly(signal, 500, fs, axis=0)

        # Convert age to float, replacing None with NaN
        age_np = np.array(age if age is not None else np.nan, dtype=np.float32)
        # sex is already a one-hot numpy array
        meta_feature_current = np.concatenate(([age_np], sex), dtype=np.float32)

        # Standardize length to 5000 points or apply sliding window
        current_length = signal.shape[0]
        window_size = 5000
        step_size = 3000 # 2000 overlap

        segments_to_process = []

        if current_length < window_size:
            # Pad with zeros if length is less than 5000
            standardized_signal = np.empty((window_size, signal.shape[1]), dtype=np.float32)
            standardized_signal[:current_length] = signal
            standardized_signal[current_length:] = 0
            segments_to_process.append(standardized_signal)
        else:
            # Apply sliding window
            for start_idx in range(0, current_length - window_size + 1, step_size):
                end_idx = start_idx + window_size
                segment = signal[start_idx:end_idx]
                segments_to_process.append(segment)
        
        # Process each segment
        for segment_signal in segments_to_process:
            # Filtering
            nyquist = 0.5 * 500
            
            # Highpass filter (1 Hz)
            highpass_cutoff = 1 / nyquist
            b, a = butter(2, highpass_cutoff, btype='high')
            segment_signal = filtfilt(b, a, segment_signal, axis=0)
            
            # Lowpass filter (30 Hz)
            lowpass_cutoff = 30 / nyquist
            b, a = butter(2, lowpass_cutoff, btype='low')
            segment_signal = filtfilt(b, a, segment_signal, axis=0)

            # Powerline noise removal (Notch filters for 50 Hz and 60 Hz)
            # 50 Hz notch
            notch_freq_50 = 50
            bandwidth_50 = 5
            freq_50 = notch_freq_50 / nyquist
            bw_50 = bandwidth_50 / nyquist
            b, a = butter(2, [freq_50 - bw_50/2, freq_50 + bw_50/2], btype='bandstop')
            segment_signal = filtfilt(b, a, segment_signal, axis=0)
            
            # 60 Hz notch
            notch_freq_60 = 60
            bandwidth_60 = 5
            freq_60 = notch_freq_60 / nyquist
            bw_60 = bandwidth_60 / nyquist
            b, a = butter(2, [freq_60 - bw_60/2, freq_60 + bw_60/2], btype='bandstop')
            segment_signal = filtfilt(b, a, segment_signal, axis=0)

            # Normalization
            segment_signal = np.ascontiguousarray(segment_signal.T) # Transpose for channel-wise normalization
            signal_mean = np.mean(segment_signal, axis=0)
            signal_std = np.std(segment_signal, axis=0)
            signal_std[signal_std < 1e-8] = 1.0 # Avoid division by zero
            segment_signal = (segment_signal - signal_mean) / signal_std
            segment_signal = np.ascontiguousarray(segment_signal.T) # Transpose back

            # Apply signal quality checks
            if contains_nan(segment_signal):
                print(f"Skipping segment from record {record_name} due to NaN values.")
                continue
            if has_zero_lead(segment_signal):
                print(f"Skipping segment from record {record_name} due to zero lead.")
                continue
            if is_low_variance(segment_signal):
                print(f"Skipping segment from record {record_name} due to low variance.")
                continue
            if not is_amplitude_normal(segment_signal):
                print(f"Skipping segment from record {record_name} due to abnormal amplitude.")
                continue

            # Check for NaN values in meta features
            if np.isnan(meta_feature_current).any():
                print(f"Skipping segment from record {record_name} due to NaN values in meta features.")
                continue

            processed_segments_data.append((segment_signal, meta_feature_current, domain_label, label))

    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
    return processed_segments_data

def main():
    parser = argparse.ArgumentParser(description="Prepare CSPC data from WFDB files.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing WFDB files (e.g., A0001.hea, A0001.mat).")
    parser.add_argument('--output_path', type=str, default="processed_CSPC_data.hdf5",
                        help="Path to the output HDF5 file.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_hdf5_path = args.output_path

    # Define standard output channels for reordering
    standard_channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels_output = len(standard_channels)

    # Find all WFDB files
    wfdb_files = find_wfdb_files(data_dir)
    print(f"Found {len(wfdb_files)} WFDB files.")

    domain_label_map = {
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

    # Determine domain_label from output_hdf5_path
    filename_without_ext = os.path.splitext(os.path.basename(output_hdf5_path))[0]
    # Extract the part before '_data' if it exists, otherwise use the whole filename
    if '_data' in filename_without_ext:
        dataset_prefix = filename_without_ext.split('_data')[0]
    else:
        dataset_prefix = filename_without_ext
    
    # Get the domain label, default to -1 if not found
    fixed_domain_label = domain_label_map.get(dataset_prefix, -1)
    print(f"Determined domain label from output path '{output_hdf5_path}': {dataset_prefix} -> {fixed_domain_label}")

    processed_count = 0
    with h5py.File(output_hdf5_path, 'w') as f:
        dset_signals = f.create_dataset('signals', shape=(0, 5000, num_channels_output), 
                                        maxshape=(None, 5000, num_channels_output), 
                                        dtype=np.float32, compression=None)
        dset_meta_features = f.create_dataset('meta_features', shape=(0, 4),
                                              maxshape=(None, 4),
                                              dtype=np.float32, compression=None)
        dset_domain_labels = f.create_dataset('domain_labels', shape=(0, 1),
                                              maxshape=(None, 1),
                                              dtype=np.float32, compression=None)
        dset_labels = f.create_dataset('labels', shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=np.float32, compression=None)

        # Use ProcessPoolExecutor for parallel processing
        # max_workers=None uses os.cpu_count()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Map the process_single_file function to all wfdb_files
            # Pass standard_channels, fixed_domain_label, and data_dir as additional arguments
            results = executor.map(process_single_file, wfdb_files, 
                                   [standard_channels] * len(wfdb_files),
                                   [fixed_domain_label] * len(wfdb_files)) # Pass the fixed domain label

            for file_segments_data in results:
                for segment_signal, meta_feature_current, domain_label, label in file_segments_data:
                    current_idx = dset_signals.shape[0]
                    dset_signals.resize(current_idx + 1, axis=0)
                    dset_signals[current_idx] = segment_signal

                    dset_meta_features.resize(current_idx + 1, axis=0)
                    dset_meta_features[current_idx] = meta_feature_current

                    dset_domain_labels.resize(current_idx + 1, axis=0)
                    dset_domain_labels[current_idx] = domain_label

                    dset_labels.resize(current_idx + 1, axis=0)
                    dset_labels[current_idx] = label
                    processed_count += 1

    print(f"Processed {processed_count} records and saved to {output_hdf5_path}")

if __name__ == "__main__":
    main()
