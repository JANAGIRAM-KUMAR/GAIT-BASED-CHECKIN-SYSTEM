import numpy as np


def extract_features(window):
    features = []
    for i in range(3):
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


def time_warp(window, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    indices = np.round(np.linspace(0, len(window) - 1, int(len(window) * scale))).astype(int)
    indices = np.clip(indices, 0, len(window) - 1)
    return window[indices]

def add_jitter(window, sigma=0.05):
    noise = np.random.normal(0, sigma, window.shape)
    return window + noise

def flip_axes(window):
    flip_mask = np.random.choice([1, -1], size=(1, window.shape[1]))
    return window * flip_mask


def segment_and_extract(accel_data, window_size=100, step_size=50, augment=True):
    segments = []
    for start in range(0, len(accel_data) - window_size, step_size):
        end = start + window_size
        window = accel_data[start:end]
        if window.shape[0] == window_size:

            segments.append(extract_features(window))

            if augment:

                segments.append(extract_features(add_jitter(window)))

                segments.append(extract_features(time_warp(window)))

                segments.append(extract_features(flip_axes(window)))

    return segments