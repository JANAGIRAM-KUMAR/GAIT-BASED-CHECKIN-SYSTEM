import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from .feature_extraction import segment_and_extract

def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def butter_highpass_filter(data, cutoff=0.3, fs=50, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def preprocess_accel_data(data, fs=50):

    data_no_gravity = butter_highpass_filter(data[:, :3], cutoff=0.3, fs=fs)

    data_filtered = butter_lowpass_filter(data_no_gravity, cutoff=5, fs=fs)

    magnitude = np.linalg.norm(data_filtered, axis=1).reshape(-1, 1)

    return np.hstack((data_filtered, magnitude))

def remove_outliers(data, threshold=3.0):
    z_scores = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    mask = np.all(np.abs(z_scores) < threshold, axis=1)
    return data[mask]

def load_all_employee_data(data_dir, sampling_rate=50, window_seconds=2, overlap=0.9):
    all_features = []
    labels = []
    window_size = int(window_seconds * sampling_rate)
    step_size = int(window_size * (1 - overlap))
    employee_label = 1

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath)

                raw_accel_data = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].values

                preprocessed_data = preprocess_accel_data(raw_accel_data, fs=sampling_rate)

                segments = segment_and_extract(preprocessed_data, window_size, step_size)
                all_features.extend(segments)
                labels.extend([employee_label] * len(segments))

        if files:
            employee_label += 1

    return pd.DataFrame(all_features), labels
