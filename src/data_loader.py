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

                accel_data = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']].values
                segments = segment_and_extract(accel_data, window_size, step_size)
                all_features.extend(segments)
                labels.extend([employee_label] * len(segments))

        # Only increment label after visiting a folder (i.e., an employee)
        if files:
            employee_label += 1

    return pd.DataFrame(all_features), labels
