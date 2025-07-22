import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import nn
from scipy.signal import butter
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from scipy import stats as st

class MyDataset_(Dataset):
    def __init__(self, data, target, label, transform=None):
        self.data = data
        self.target = target
        self.label = label
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        labl = self.label[index]
        return x, y, labl
    def __len__(self):
        return len(self.data)

def rpeak_detection(ecg_data, sampling_rate=100.):
    # rpeak segmentation for ori_ecg
    rpeaks, = ecg.hamilton_segmenter(
        signal=ecg_data, sampling_rate=sampling_rate)
    rpeaks, = ecg.correct_rpeaks(
        signal=ecg_data,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        tol=0.05)
    return rpeaks

class ECGDataset(Dataset):
    def __init__(self,  data, target, label, sigma=1., scaled_exp=True):
        self.data = data
        self.target = target
        self.label = label
        self.ws = 256  # training window size
        self.all_ws = 512  # full window size
        self.sampling_rate = 125.
        self.sigma = sigma
        self.scaled_exp = scaled_exp

    def expand_rpeaks(self, rpeaks):
        if len(rpeaks) == 0:
            return np.zeros(self.all_ws), np.zeros(self.all_ws)
        normal_rpeaks = np.zeros(self.all_ws)
        onehot_rpeaks = np.zeros(self.all_ws)
        new_rpeaks = np.zeros(self.all_ws)
        for rpeak in rpeaks:
            nor_rpeaks = st.norm.pdf(
                np.arange(0, self.all_ws), loc=rpeak, scale=self.sigma)
            normal_rpeaks = normal_rpeaks + nor_rpeaks
            st_ix = np.clip(
                rpeak-50//(1000/self.sampling_rate), 0, self.all_ws)
            ed_ix = np.clip(
                rpeak+70//(1000/self.sampling_rate), 0, self.all_ws)
            st_ix = np.int64(st_ix)
            ed_ix = np.int64(ed_ix)
            onehot_rpeaks[st_ix:ed_ix] = 1
            
        new_rpeaks[:len(rpeaks)] = rpeaks[:]
        # scale to [0, 1]
        if self.scaled_exp:
            normal_rpeaks = (normal_rpeaks - np.min(normal_rpeaks))
            normal_rpeaks = normal_rpeaks/np.ptp(normal_rpeaks)
        return np.array(normal_rpeaks), np.array(onehot_rpeaks), np.array(new_rpeaks)

    def __getitem__(self, idx: int):
        # rpeak detection
        ppg = self.data[idx]
        ecg = self.target[idx]
        labl = self.label[idx]
        rpeaks = rpeak_detection(ecg, self.sampling_rate)
        
        normal_rpeaks, onehot_rpeaks, new_rpeaks = self.expand_rpeaks(rpeaks)
        exp_rpeaks = torch.from_numpy(
            normal_rpeaks.reshape(1, -1)).type(torch.FloatTensor)
        rpeaks_arr = torch.from_numpy(
            onehot_rpeaks.reshape(1, -1)).type(torch.LongTensor)
        rpeaks_new = torch.from_numpy(
            new_rpeaks.reshape(1, -1)).type(torch.LongTensor)        
        
        return {'ppg': ppg,
                'ecg': ecg,
                'label':labl,
                'rpeaks': rpeaks_new,
                'exp_rpeaks': exp_rpeaks}
    def __len__(self):
        return len(self.data)

class QRSLoss(nn.Module):
    def __init__(self, beta=1):
        super(QRSLoss, self).__init__()
        self.beta = beta

    def forward(self, input, target, exp_rpeaks):
        loss = F.l1_loss(
            input*(1+self.beta*exp_rpeaks), target*(1+self.beta*exp_rpeaks))
        return loss
    
def plot_losses(path, trainer):
    plt.figure(figsize=(12, 5))
    plt.title("Errors in Training")
    plt.plot(trainer.GE_errors, label='ECG Generator')
    plt.plot(trainer.GP_errors, label='PPG Generator')
    plt.plot(trainer.DEtime_errors, label='ECG Discrim. (time)')
    plt.plot(trainer.DEfreq_errors, label='ECG Discrim. (freq.)')
    plt.plot(trainer.DPtime_errors, label='PPG Discrim. (time)')
    plt.plot(trainer.DPfreq_errors, label='PPG Discrim. (freq.)')  
    plt.plot(trainer.test_GE_errors, label='ECG Generator (testing)')
    plt.plot(trainer.test_GP_errors, label='PPG Generator (testing)')  
    
    plt.xlabel("Epochs")
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(path+'Losses.png', facecolor='w', edgecolor='w', format='png',
            transparent=False, bbox_inches='tight', pad_inches=0.1)
        
def min_max_normalize(signal):
    min_value = np.min(signal)
    max_value = np.max(signal)
    normalized_signal = (signal - min_value) / (max_value - min_value)
    normalized_signal = normalized_signal * 2 - 1
    return normalized_signal

def calculate_PRD(orig_sig, recon_sig):
    num = np.sum((orig_sig - recon_sig)**2)
    den = np.sum(orig_sig**2)
    PRD = np.sqrt(num/den)
    return PRD*100   

def calculate_rRMSE(orig_sig, recon_sig):
    rrmse = np.linalg.norm(orig_sig - recon_sig)/np.linalg.norm(orig_sig)
    return rrmse

def calculate_RMSE(orig_sig, recon_sig):
    rmse = np.sqrt(np.sum((orig_sig - recon_sig)**2)/len(orig_sig))
    return rmse

def overlap(x_data, y_data):
    x = []
    y = []
    n_lap = 2
    ws = 4
    for n in range(int(len(x_data)/2)-1):
        x.append(x_data[n*(ws-n_lap):n*(ws-n_lap)+ws])
        y.append(y_data[n*(ws-n_lap):n*(ws-n_lap)+ws])
    x = np.asarray(x)
    y = np.asarray(y)
    return x.reshape((-1, ws*x.shape[-1])), y.reshape((-1, ws*x.shape[-1]))
   
def de_overlap(data, seg_len):
    n_lap = 2
    ws = 4
    data = data.reshape((-1, seg_len*ws))
    return data[:,:seg_len*(ws-n_lap)].flatten()

def smoothl1_loss(signal1, signal2):
    loss = nn.SmoothL1Loss(reduction='mean')(signal1, signal2)
    return loss

def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)       

def butter_bandpass(lowcut, highcut,fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a  
