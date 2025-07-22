import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from utils import min_max_normalize, butter_bandpass
import yaml

def remove_noise(ecg, ppg, times, peaks, intervs):
    ecg = ecg.copy()
    ppg = ppg.copy()
    times = times.copy()

    for interv in intervs:
        indices = [i for i, p in enumerate(peaks) if interv[0] < p < interv[1]]
        if not indices:
            continue
        start = peaks[max(min(indices)-1, 0)]
        end = peaks[min(max(indices)+1, len(peaks)-1)]
        ecg[start:end] = np.nan
        ppg[start:end] = np.nan
        times[start:end] = np.nan

    mask = ~np.isnan(ecg) & ~np.isnan(ppg)
    return ecg[mask], ppg[mask], times[mask]


def synchronize_real(ecg_org, ppg_org, times, name, config_path="preprocessing_config.yaml"):
    ecg_org = np.asarray(ecg_org)
    ppg_org = np.asarray(ppg_org)

    # Load subject-specific config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    subject_cfg = cfg["subjects"].get(name, {})

    # Get peak detection distance
    peak_distance = subject_cfg.get("peak_distance", 50)
    peaks_ecg0, _ = find_peaks(ecg_org, distance=peak_distance)
    peaks_ppg0, _ = find_peaks(ppg_org, distance=peak_distance)

    # Apply noise removal if intervals exist
    intervs = subject_cfg.get("intervs", [])
    if intervs:
        ecg_org, ppg_org, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)

    # Detect peaks on cleaned data
    peaks_ecg, _ = find_peaks(ecg_org, distance=peak_distance)
    peaks_ppg, _ = find_peaks(ppg_org, distance=peak_distance)

    # Apply synchronization index
    sync = subject_cfg.get("sync", [0, 1])
    trim_end = subject_cfg.get("trim_end", None)

    if sync is None:
        ecg_syn = ecg_org.copy()
        ppg_syn = ppg_org.copy()
        new_times = times.copy()
    else:
        i_ecg, i_ppg = sync
        ecg_syn = ecg_org[peaks_ecg[i_ecg]:].copy()
        ppg_syn = ppg_org[peaks_ppg[i_ppg]:].copy()
        new_times = times[peaks_ecg[i_ecg]:].copy()

    end = trim_end if trim_end else min(len(ecg_syn), len(ppg_syn))
    ecg_syn = ecg_syn[:end]
    ppg_syn = ppg_syn[:end]
    new_times = new_times[:end]

    # Visualization
    plt.figure(figsize=(10, 4))
    n = np.arange(0, 2000)
    epk = peaks_ecg[peaks_ecg < 2000]
    ppk = peaks_ppg[peaks_ppg < 2000]
    plt.scatter(epk, ecg_org[epk], marker='*')
    plt.plot(n, ecg_org[:2000], label='real ecg', color='red')
    plt.scatter(ppk, ppg_org[ppk], marker='o')
    plt.plot(n, ppg_org[:2000], label='real ppg', color='blue')
    plt.title(f"Real data (signal {name})")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(ecg_syn[:2000], label='synchronized real ecg', color='red')
    plt.plot(ppg_syn[:2000], label='synchronized real ppg', color='blue')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
    plt.close()
    return ecg_syn, ppg_syn, new_times

def get_segments(ecg_list, ppg_list, name=None, window=200):
    ecg_seg_all = []
    ppg_seg_all = []
    labels_all = []

    for i, (ecg, ppg) in enumerate(zip(ecg_list, ppg_list)):
        num_seg = len(ecg) // window
        ecg_segments = []
        ppg_segments = []

        for j in range(num_seg - 1):  # drop last incomplete
            ecg_seg = min_max_normalize(ecg[j * window:(j + 1) * window])
            ppg_seg = min_max_normalize(ppg[j * window:(j + 1) * window])

            if not np.isnan(ecg_seg).any() and not np.isnan(ppg_seg).any():
                ecg_segments.append(ecg_seg)
                ppg_segments.append(ppg_seg)

        ecg_segments = np.array(ecg_segments)
        ppg_segments = np.array(ppg_segments)

        if ecg_segments.shape[0] == 0:
            continue  # skip empty case

        print(f"[{name or i}] ecg_segments.shape = {ecg_segments.shape}")

        ecg_seg_all.append(ecg_segments)
        ppg_seg_all.append(ppg_segments)

        # Labeling
        if name is not None:
            labels_all.append(np.full(ecg_segments.shape[0], int(name) + 1000))
        else:
            labels_all.append(np.full(ecg_segments.shape[0], i))

    return (
        np.array(ecg_seg_all, dtype=object),
        np.array(ppg_seg_all, dtype=object),
        np.array(labels_all, dtype=object)
        )

if __name__ == "__main__":
    # Define subject list
    name_ids1 = ['07','08','09','16','22','30','34','37','42','43','50','51']
    name_ids2 = ['01','02','05','11','18','19','20','21','24','29','46']
    name_ids3 = ['12','14','17','27','35','38','40','47','48']
    name_ids = name_ids1 + name_ids2 + name_ids3
    
    # Global settings
    seg_len = 128
    org_samp_rate = 125
    data_dir = './data/real_data/bidmc/csv'
    out_root = './data/bidmc/processed'

    for name_id in name_ids:
        # Prepare file paths
        save_path = os.path.join(out_root, name_id, f"seg{seg_len}")
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(data_dir, f"bidmc_{name_id}_Signals.csv")

        # Load CSV and extract raw signals
        df = pd.read_csv(filename)
        times = df['Time [s]'].values
        ecg_raw = df[' II'].values
        ppg_raw = df[' PLETH'].values

        # Detrend + Bandpass filter
        ecg = signal.detrend(ecg_raw)
        b, a = butter_bandpass(lowcut=0.4, highcut=45, fs=org_samp_rate, order=6)
        ecg = signal.filtfilt(b, a, ecg)

        ppg = signal.detrend(ppg_raw)
        b, a = butter_bandpass(lowcut=0.3, highcut=8, fs=org_samp_rate, order=3)
        ppg = signal.filtfilt(b, a, ppg)

        # Synchronize and clean
        ecg, ppg, times = synchronize_real(ecg, ppg, times, name_id)

        # Segment normalized windows
        ecg_seg, ppg_seg, labels = get_segments([ecg], [ppg], name=name_id, window=seg_len)
        print(f"[{name_id}] ecg_seg.shape = {ecg_seg.shape}, ppg_seg.shape = {ppg_seg.shape}, labels.shape = {labels.shape}")

        # # Optional: Save to .npy
        np.save(os.path.join(save_path, 'real_ecg.npy'), ecg_seg)
        np.save(os.path.join(save_path, 'real_ppg.npy'), ppg_seg)
        np.save(os.path.join(save_path, 'labels.npy'), labels)
        np.save(os.path.join(save_path, 'times.npy'), times)
