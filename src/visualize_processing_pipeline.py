import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.feature_extraction import segment_and_extract

def high_pass_filter(data, cutoff=0.3, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def visualize_single_csv(csv_path, sampling_rate=50, window_seconds=2, overlap=0.5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from src.feature_extraction import segment_and_extract
    from scipy.signal import butter, filtfilt

    def high_pass_filter(data, cutoff=0.3, fs=50, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    df = pd.read_csv(csv_path)
    time = np.arange(len(df)) / sampling_rate

    ax, ay, az = df['ax (m/s^2)'], df['ay (m/s^2)'], df['az (m/s^2)']
    ax_f, ay_f, az_f = high_pass_filter(ax), high_pass_filter(ay), high_pass_filter(az)

    ax_g, ay_g, az_g = ax - ax_f, ay - ay_f, az - az_f

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (raw, filt, grav, label) in enumerate(zip([ax, ay, az], [ax_f, ay_f, az_f], [ax_g, ay_g, az_g], ['X', 'Y', 'Z'])):
        axs[i].plot(time, raw, label=f'{label} Raw', alpha=0.4)
        axs[i].plot(time, filt, label=f'{label} Filtered', linewidth=1.5)
        axs[i].plot(time, grav, label=f'{label} Gravity Component', linestyle='--')
        axs[i].set_ylabel('Acc (m/s²)')
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle("Gravity Removal via High-Pass Filter (Cutoff: 0.3 Hz)", fontsize=16)
    plt.tight_layout()
    plt.show()

    accel_data = np.stack([ax_f, ay_f, az_f], axis=1)
    window_size = int(window_seconds * sampling_rate)
    step_size = int(window_size * (1 - overlap))
    segments = segment_and_extract(accel_data, window_size, step_size)

    print(f"📦 Segments Extracted: {len(segments)}")
    print(f"🧬 Features per Segment: {len(segments[0])}")

