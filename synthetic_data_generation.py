import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq 
import pandas as pd
from scipy.integrate import odeint
from utils import butter_bandpass



def synchronize_data(ecg_real, ppg_real, ecg_syn, ppg_syn, start_real, path, name, paras, paras_name):
    ecg_synch_real = ecg_real[start_real:].copy()
    ppg_synch_real = ppg_real[start_real:].copy()
    
    ecg_synch_synth = ecg_syn.copy()
    ppg_synch_synth = ppg_syn.copy()
                      
    return ecg_synch_real, ppg_synch_real, ecg_synch_synth, ppg_synch_synth

def synchronize_synth(ecg, ppg, start_ecg, start_ppg):
    ecg = np.asarray(ecg)
    ppg = np.asarray(ppg)

    ecg_peaks, _ = find_peaks(ecg, distance=50)
    ppg_peaks, _ = find_peaks(ppg, distance=50)
        
    ecg_syn = ecg[start_ecg:].copy()
    ppg_syn = ppg[start_ppg:].copy()
    end = min(len(ecg_syn), len(ppg_syn))

    return ecg_syn[:end], ppg_syn[:end]

def synchronize_real(ecg_org, ppg_org):
    ecg = np.asarray(ecg_org)
    ppg = np.asarray(ppg_org)

    peaks_ecg, _ = find_peaks(ecg, distance=50)
    peaks_ppg, _ = find_peaks(ppg, distance=50)

    ecg_syn = ecg[peaks_ecg[0]:].copy()
    ppg_syn = ppg[peaks_ppg[0]:].copy()
    ecg_peaks = peaks_ecg[0:] - peaks_ecg[0]
    end = min(len(ecg_syn), len(ppg_syn))  

    plt.figure(figsize=(10,4))
    n = np.arange(0, 2000)
    epk = peaks_ecg[peaks_ecg < 2000]
    ppk = peaks_ppg[peaks_ppg < 2000]
    plt.scatter(epk, ecg[epk], marker='*')
    plt.plot(n, ecg[:2000],label = 'real ecg', color='red')
    plt.scatter(ppk, ppg[ppk], marker='o')
    plt.plot(n, ppg[:2000], label = 'real ppg', color='blue')
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
        
    return ecg_[:end], ppg_[:end], ecg_peaks[ecg_peaks<end]

def generate_random(para, ind, percent_std, percent):
    new_para = para.copy()
    for i in range(len(para)):
        if i not in [ind - 1, ind, ind + 1]:
            level = np.random.normal(0, percent_std * abs(new_para[i]))
            level = np.clip(level, -percent * abs(new_para[i]), percent * abs(new_para[i]))
            new_para[i] += level
    return new_para

def get_parameters(A, B, Theta, std, pct, flag=False):
    ai = A.copy()
    bi = B.copy()
    thei = Theta.copy()
    inds = np.where(thei == 0)[0]

    if flag:
        ai = generate_random(ai, inds, std, pct)
        bi = generate_random(bi, inds, std, pct)
        thei = generate_random(thei, inds, std, pct)

    return ai, bi, thei

  
def calculate_RR_real(ecg_peaks):
    RR = []
    for n in range(len(ecg_peaks)-1):
        d1 = ecg_peaks[n+1] - ecg_peaks[n]
        RR.append(d1)
    RR = np.array(RR)
    return RR

def calculate_RR_synthtic(ecg_synth, ecg_peaks):
    RR_syn = []
    for n in range(len(ecg_peaks)-1): ####### range(len(ecg_peaks)-1)
        if n == 0:
            d1 = ecg_peaks[n] - 0
            RR_syn.append(d1)
        else:
            d1 = ecg_peaks[n+1] - ecg_peaks[n]
            RR_syn.append(d1)
    RR_syn = np.asarray(RR_syn[1:])
    
    return RR_syn
    
# def plot_power_spectrum(ecg_real, ecg_synth, path, name_id):
#     ### plot power spectrum
#     ws = 150
#     seg_real = ecg_real[:ws]
#     N = int(ws)
    
#     power_spectrum_real = np.abs(rfft(seg_real))
#     freqs_real = rfftfreq(N, 1.0/org_samp_rate)
#     seg_synth = ecg_synth[:ws]
#     power_spectrum_synth = np.abs(rfft(seg_synth))
#     freqs_synth = rfftfreq(N, 1.0/org_samp_rate)

#     fig, axs = plt.subplots(4, 1, figsize=(10, 15))
#     axs[0].plot(seg_real,  color='red', label='real ECG')
#     axs[0].set_title("Real ECG",  fontsize = 10)
#     axs[0].grid(True)
#     axs[1].plot(seg_synth,  color='green', label='Synthetic ECG')
#     axs[1].set_title("Synthetic ECG",  fontsize = 10)
#     axs[1].grid(True)
#     axs[2].plot(freqs_real, power_spectrum_real, color='red')
#     axs[2].set_title("Power spectrum of real ECG",  fontsize = 10)
#     axs[2].grid(True)
#     axs[3].plot(freqs_synth, power_spectrum_synth, color='green')
#     axs[3].set_title("Power spectrum of synthetic ECG",  fontsize = 10)
#     axs[3].grid(True)
#     fig.suptitle(f'Signal {name_id}')
#     plt.legend(loc="upper left")
#     # plt.savefig(path+f"Power spectrum_{name_id}.png")
#     plt.show()
#     plt.close()
    
def ppgeq(u, t, *args):
    global A, B, f

    ai = args[0]
    bi = args[1]
    thei = args[2]
    freqs = args[3]
    time_int = args[4]
    x, y, z, v, w = u
    alp = 1 - np.sqrt(x**2 + y**2)
    the = np.arctan2(y, x)

    int_order=np.digitize(t, time_int)
    ome=2 * np.pi * freqs[int_order-1]
    the = f[0]/freqs[int_order-1]*the
    
    dthe = the - thei
    z0 = A * np.sin(2 * np.pi * f[1] * t)
    temp = -np.dot(ai.T, dthe * np.exp(-dthe**2 / bi**2 / 2))
    udot = np.zeros(5)
    udot[0] = alp * x - ome * y
    udot[1] = alp * y + ome * x
    udot[2] = temp - (z - z0)
    udot[3] = -B[0] * v + B[1] * w 
    udot[4] = z**2 - B[2] * w
    
    return udot

if __name__=="__main__":  

    # Load configuration
    with open('simulation_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    org_samp_rate = cfg['sampling']['org_rate']
    
    # Filter parameters
    ecg_filter = cfg['filters']['ecg']
    ppg_filter = cfg['filters']['ppg']
    
    # ODE parameters
    A = cfg['ODE']['A']
    B = np.array(cfg['ODE']['B'])
    f = np.array(cfg['ODE']['f'])
    thei = np.array(cfg['ODE']['thei'])
    ai = np.array(cfg['ODE']['ai'])
    bi = np.array(cfg['ODE']['bi'])
    u0 = np.array(cfg['ODE']['u0'])
    beatPerSec = cfg['ODE']['beatPerSec']
    
    random_std = cfg['ODE']['randomization']['std']
    random_pct = cfg['ODE']['randomization']['percent']
    
    input_file = cfg['filepaths']['input']
    df = pd.read_csv(input_file)
    
    # Access the data in a specific column by name
    time = df['Time [s]'].values
    ecg = df[' II'].values
    ppg = df[' PLETH'].values
    
    ecg = signal.detrend(ecg)
    b,a = butter_bandpass(lowcut=0.4, highcut=15, fs=org_samp_rate, order=6)
    ecg = scipy.signal.filtfilt(b, a, ecg)

    ppg = signal.detrend(ppg)
    b,a = butter_bandpass(lowcut=0.3, highcut=8, fs=org_samp_rate, order=3)
    ppg = scipy.signal.filtfilt(b, a, ppg)
    
    ecg, ppg, peaks_real = synchronize_real(ecg, ppg) 
    RR = calculate_RR_real(peaks_real)
            
    mean_RR = np.mean(RR)
    std_RR = np.std(RR)
    if std_RR > 1:
        mean_RR = mean_RR + std_RR/2

    num_beats = len(RR)
    T = num_beats * beatPerSec ## 60 beats, 10s per beat
    pointsPerBeat = round(mean_RR)
    
    NT = int(pointsPerBeat * num_beats)
    t = np.linspace(0, T, NT)
    freqs = f[0]*round(mean_RR)/RR
    time_ = np.arange(num_beats)*pointsPerBeat 
 
    for i in range(len(time_)-1):
        if i > 0:
            dif = abs(round(mean_RR) - RR[i])
            if round(mean_RR) - RR[i] > 0:
                time_[i+1:] =  time_[i+1:] - dif 
            if round(mean_RR) - RR[i] < 0 and (time_[i+1:] + dif )[-1] <= (len(t)-1):
                time_[i+1:] =  time_[i+1:] + dif
              
  
    time_interv = t[time_]   
    ai_new, bi_new, thei_new = get_parameters(ai, bi, thei, random_std, random_pct, flag=True)
    
    def simulate_ode_ecg_ppg(u0, t, ai, bi, thei, freqs, time_interv):
        u = odeint(ppgeq, u0, t, args=(ai, bi, thei, freqs, time_interv))
        x = u[:, 0]
        y = u[:, 1]
        z = u[:, 2]
        v = u[:, 3]
        return z, v  # ecg, ppg

    ecg_syn, ppg_syn = simulate_ode_ecg_ppg(u0, t, ai_new, bi_new, thei_new, freqs, time_interv)

    ecg_peaks_syn, _ = find_peaks(ecg_syn, distance=50)
    ppg_peaks_syn, _ = find_peaks(ppg_syn, distance=50)

    RR_syn = calculate_RR_synthtic(ecg_syn, ecg_peaks_syn)
    mean_RR_syn = np.mean(RR_syn)
    std_RR_syn = np.std(RR_syn)
        
    start_syn_ecg = ecg_peaks_syn[1]
    start_syn_ppg = ppg_peaks_syn[1]
    start_real2 = peaks_real[1]

    ecg_synch_synth, ppg_synch_synth = synchronize_synth(ecg_syn, ppg_syn, start_syn_ecg, start_syn_ppg)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot ECG
    axs[0].plot(ecg_synch_synth[:500], color='red')
    axs[0].set_title('Synthetic ECG')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    # Plot PPG
    axs[1].plot(ppg_synch_synth[:500], color='blue')
    axs[1].set_title('Synthetic PPG')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
        
        