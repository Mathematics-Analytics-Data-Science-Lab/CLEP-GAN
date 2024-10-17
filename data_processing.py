# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt


# Some Functions
def butter_bandpass(lowcut, highcut,fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a 

def remove_noise(ecg_syn, ppg_syn, times, peaks, intervs):
    for interv in intervs:
        indices = [i for i, p in enumerate(peaks) if  interv[0] < p < interv[1]]
        print(interv)
        start = peaks[min(indices)-1]
        end = peaks[max(indices)+1]
        ecg_syn[start:end] = np.nan
        ppg_syn[start:end] = np.nan
        times[start:end] = np.nan
  
    clean_ecg = ecg_syn[~np.isnan(ecg_syn)]
    clean_ppg = ppg_syn[~np.isnan(ppg_syn)]
    new_times = times[~np.isnan(times)]
    return clean_ecg, clean_ppg, new_times
    
    
def synchronize_real(ecg_org, ppg_org, times, name):

    ecg_org = np.asarray(ecg_org)
    ppg_org = np.asarray(ppg_org)
    
    if name =='05':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[14650, 20000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)   
    elif name == '12':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[24000, 27000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)   
    elif name == '16':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[14250,15250], [24000,26000], [31700, 34000],[35250,35500], [46000,46300]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name == '21':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[16750, 20000], [21550, 22100], [28750, 36000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name == '22':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[4000, 6000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name =='29':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[11400,13600], [16600, 18200],[19800,20150], [27800,28600],[39000,40000],[43600,48000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)  
    elif name == '30':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[12750,13400], [20900,21200], [21700, 21900], [27100,27500],[35250,35650],[37370,37700],[46870,48000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 

    elif name == '34':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[10950, 13050,], [15900, 16150], [48000, 48700], [51450, 52000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name == '37':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[4000, 6000], [46500, 48000], [53000, 56000]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)  
    elif name == '43':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=60)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=60)
        intervs = [[9400,10600], [15500,15900], [19300, 19900], [36800, 37300]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name =='46':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=60)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=60)
        intervs = [[11000,15200], [22000,24000], [26800, 28000], [33600, 35400]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs) 
    elif name == '50':
        peaks_ecg0, _ = find_peaks(ecg_org, distance=50)
        peaks_ppg0, _ = find_peaks(ppg_org, distance=50)
        intervs = [[9000, 9500], [29250, 32000], [40000, 42750]]
        ecg, ppg, times = remove_noise(ecg_org, ppg_org, times, peaks_ecg0, intervs)   

    else:
        ecg = ecg_org
        ppg = ppg_org

    if name in ['43','36']:
        peaks_ecg, _ = find_peaks(ecg, distance=60)
        peaks_ppg, _ = find_peaks(ppg, distance=60)
    else:
        peaks_ecg, _ = find_peaks(ecg, distance=50)
        peaks_ppg, _ = find_peaks(ppg, distance=50)
    
 
    ## synchronization
    if name in ['07']:
        ecg_syn = ecg[peaks_ecg[0]:].copy()
        ppg_syn = ppg[peaks_ppg[0]:].copy()
        new_times = times[peaks_ecg[0]:].copy()
        end = min(len(ecg_syn), len(ppg_syn))  
    elif name in ['09','11','17','19','35','37','40','41','42', '43', '51']:
        ecg_syn = ecg[peaks_ecg[1]:].copy()
        ppg_syn = ppg[peaks_ppg[1]:].copy()
        new_times = times[peaks_ecg[1]:].copy()
        end = min(len(ecg_syn), len(ppg_syn))   
    elif name =='29':
        ecg_syn = ecg[peaks_ecg[0]:].copy()
        ppg_syn = ppg[peaks_ppg[1]:].copy()
        new_times = times[peaks_ecg[0]:].copy()
        end = 57000  
    elif name == '30':
        ecg_syn = ecg[peaks_ecg[1]:].copy()
        ppg_syn = ppg[peaks_ppg[1]:].copy()
        new_times = times[peaks_ecg[1]:].copy()
        end = 51100 
    elif name =='34':
        ecg_syn = ecg[peaks_ecg[1]:].copy()
        ppg_syn = ppg[peaks_ppg[1]:].copy()
        new_times = times[peaks_ecg[1]:].copy()
        end = 54000  
    elif name == '49':
        ecg_syn = ecg.copy()
        ppg_syn = ppg.copy()
        new_times = times.copy()
        end = min(len(ecg_syn), len(ppg_syn)) 
    elif name == '38':
        ecg_syn = ecg[peaks_ecg[1]:].copy()
        ppg_syn = ppg[peaks_ppg[2]:].copy()
        new_times = times[peaks_ecg[1]:].copy()
        end = min(len(ecg_syn), len(ppg_syn))
    else:
        ecg_syn = ecg[peaks_ecg[0]:].copy()
        ppg_syn = ppg[peaks_ppg[1]:].copy()
        new_times = times[peaks_ecg[0]:].copy()
        end = min(len(ecg_syn), len(ppg_syn))
    
    
    plt.figure(figsize=(10,4))
    n = np.arange(0, 2000)
    epk = peaks_ecg[peaks_ecg < 2000]
    ppk = peaks_ppg[peaks_ppg < 2000]
    plt.scatter(epk, ecg[epk], marker='*')
    plt.plot(n, ecg[:2000],label = 'real ecg', color='red')
    plt.scatter(ppk, ppg[ppk], marker='o')
    plt.plot(n, ppg[:2000], label = 'real ppg', color='blue')
    plt.title(f"Real data (signal {name})")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,4))
    plt.plot(ecg_syn[0:2000],label = 'synchronized real ecg', color='red')
    plt.plot(ppg_syn[0:2000], label = 'synchronized real ppg', color='blue')

    plt.legend(loc="lower left")
    plt.show()
    plt.close()
    
   
    ecg_ = ecg_syn
    ppg_ = ppg_syn
    

    return ecg_[:end], ppg_[:end], new_times[:end]

def min_max_normalize(signal):
    min_value = np.min(signal)
    max_value = np.max(signal)
    normalized_signal = (signal - min_value) / (max_value - min_value)
    normalized_signal = normalized_signal * 2 - 1

    return normalized_signal

def get_segments(ecg, ppg, name=None, window=200):
    ecg_seg_all = []
    ppg_seg_all = []
    labels_all = []
    for i in range(len(ecg)):
        num_seg = int(len(ecg[i])/window)
        ecg_segments = []
        ppg_segments = []
        labels = []
        
        for j in range(num_seg-1):
            ecg_seg = ecg[i][j*window:(j+1)*window]
            ecg_seg = min_max_normalize(ecg_seg)
            ppg_seg = ppg[i][j*window:(j+1)*window]
            ppg_seg = min_max_normalize(ppg_seg)
            
            if np.isnan(ecg_seg).any() or np.isnan(ppg_seg).any():
                continue
            else:
                ecg_segments.append(ecg_seg)
                ppg_segments.append(ppg_seg)
            
                
        ecg_segments = np.asarray(ecg_segments)
        ppg_segments = np.asarray(ppg_segments)
        print("ecg_segments.shape=",ecg_segments.shape)
        
        ecg_seg_all.append(ecg_segments)
        ppg_seg_all.append(ppg_segments)
        if name:
            labels_all.append(np.asarray([int(name_id)+1000]*len(ecg_segments)))
        else:
            labels_all.append(np.asarray([i]*len(ecg_segments)))
        
            
    return np.asarray(ecg_seg_all), np.asarray(ppg_seg_all), np.asarray(labels_all)


if __name__=="__main__":  
    name_ids1 = ['07','08','09','16','22','30','34','37','42','43','50','51']
    name_ids2 = ['01','02','05','11','18','19','20','21','24','29','46']
    name_ids3 = ['12','14','17','27','35','38','40','47','48']
    name_ids = name_ids1 + name_ids2 + name_ids3

    seg_len = 128
    for name_id in name_ids:

        save_path = f"./data/bidmc/processed/{name_id}/seg{seg_len}"
        org_samp_rate = 125
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    
        filename = f'./data/bidmc/csv/bidmc_{name_id}_Signals.csv'
        df = pd.read_csv(filename)
        
        # Access the data in a specific column by name
        times = df['Time [s]'].values
        
        ecg = df[' II'].values
        ecg = signal.detrend(ecg)
        b,a = butter_bandpass(lowcut=0.4, highcut=45, fs=125, order=6)
        ecg = scipy.signal.filtfilt(b, a, ecg)
        # print(ecg)
        ppg = df[' PLETH'].values
        ppg = signal.detrend(ppg)
        b,a = butter_bandpass(lowcut=0.3, highcut=8, fs=125, order=3)
        ppg = scipy.signal.filtfilt(b, a, ppg)
        
     
        #### Visualization for real data
    
        # ecg, ppg, times = clean_signals(ecg, ppg, times, path, name=name_id) 
        ecg, ppg, times = synchronize_real(ecg, ppg, times, name_id)
     
        ecg_seg_real, ppg_seg_real, labels_real = get_segments([ecg], [ppg], name=name_id, window=seg_len)
        print('ecg_seg_real.shape=',ecg_seg_real.shape)
        print('ppg_seg_real.shape=',ppg_seg_real.shape)
        print('labels_real.shape=',labels_real.shape)

    
        # np.save(os.path.join(save_path, 'real_ecg.npy'), ecg_seg_real)
        # np.save(os.path.join(save_path, 'real_ppg.npy'), ppg_seg_real)
        # np.save(os.path.join(save_path, 'labels_real.npy'), labels_real)
        # np.save(os.path.join(save_path, 'times.npy'), times)
        