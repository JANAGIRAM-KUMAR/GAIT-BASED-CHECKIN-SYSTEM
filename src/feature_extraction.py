import numpy as np

def extract_features(window):
    features = []
    for i in range(window.shape[1]):
        axis = window[:, i]
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.min(axis),
            np.max(axis),
            np.max(axis) - np.min(axis),
            np.sum(np.square(axis)),
        ])
    sma = np.mean(np.sum(np.abs(window[:, :3]), axis=1))
    features.append(sma)
    return features

def segment_and_extract(accel_data, window_size=100, step_size=50):
    segments = []
    for start in range(0, len(accel_data) - window_size, step_size):
        end = start + window_size
        window = accel_data[start:end]
        if window.shape[0] == window_size:
            features = extract_features(window)
            segments.append(features)
    return segments
