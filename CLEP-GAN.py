#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import AdamW, Adam
from utils import *
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from unet import *

    
def test(args, trainer, test_loader, signal_size, epoch):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netGE_chk = torch.load(args.chk_pth+f"generator_ecg{epoch}.pth")
    netGP_chk = torch.load(args.chk_pth+f"generator_ppg{epoch}.pth")
    
    result_path = args.results 
    trainer.netGE.load_state_dict(netGE_chk) 
    trainer.netGP.load_state_dict(netGP_chk)
    
    ids = list(signal_size.keys())
    generate_ecg = []
    recon_ecg2ecg = []
    recon_ppg2ppg = []
    ppg_sig = []
    ecg_sig = []
    rmse_ecg = []
    rrmse_ecg = []
    start_time = time.time()
    for i, batch in enumerate(test_loader, 0):
        real_ppg = batch['ppg']      
        real_ecg = batch['ecg']

        gen_ppg2ppg, connections, _ = trainer.netGP(real_ppg.to(device))
        gen_ecg, _  = trainer.netGE.generate(connections)
        
        
        gen_ecg2ecg, connections, _  = trainer.netGE(real_ecg.to(device))
        
        
        gen_ecg = gen_ecg.detach().cpu().squeeze(1).numpy()
        gen_ecg2ecg = gen_ecg2ecg.detach().cpu().squeeze(1).numpy()
        gen_ppg2ppg = gen_ppg2ppg.detach().cpu().squeeze(1).numpy()
        
        ppg_sig.extend(real_ppg)
        ecg_sig.extend(real_ecg)
        generate_ecg.extend(gen_ecg)
        recon_ppg2ppg.extend(gen_ppg2ppg)
        recon_ecg2ecg.extend(gen_ecg2ecg)
        
    
    ############### save each testing signal###################################
    ecg_dic = {i: [] for i in ids}
    generate_ecg_dic = {i: [] for i in ids}
    ppg_dic = {i: [] for i in ids}
    recon_ppg2ppg_dic = {i: [] for i in ids}
    recon_ecg2ecg_dic = {i: [] for i in ids}
    pre_num = 0 
    for i in range(len(ids)):
        idx = ids[i]
        num = signal_size[idx]
        for j in range(len(generate_ecg)):
            if j in range(pre_num, pre_num+num):
                ppg_dic[idx].extend(ppg_sig[j])
                recon_ppg2ppg_dic[idx].extend(recon_ppg2ppg[j]) 
                ecg_dic[idx].extend(ecg_sig[j]) 
                generate_ecg_dic[idx].extend(generate_ecg[j])
                recon_ecg2ecg_dic[idx].extend(recon_ecg2ecg[j]) 
                 
        pre_num += num
                                
        img_path = result_path + 'generated/plots/{}/'.format(idx)
        if not os.path.exists(img_path):
            os.makedirs(img_path )

    ############### write resutls to files ###################################        
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    results_f = result_path + 'results.txt'
    sig_path = result_path + 'generated/signal/'
    if not os.path.exists(sig_path):
        os.makedirs(sig_path )
        
    prd_ecg_dic = {}
    rmse_ecg_dic = {}
    rrmse_ecg_dic = {}

    for idx in ids:        
        gene_ecg_signal = np.asarray(generate_ecg_dic[idx])
        orig_ecg_signal = np.asarray(ecg_dic[idx])
        recon_ecg2ecg_signal = np.asarray(recon_ecg2ecg_dic[idx])
        recon_ppg2ppg_signal = np.asarray(recon_ppg2ppg_dic[idx])
        orig_ppg_signal = np.asarray(ppg_dic[idx])

        prd_ecg = np.sqrt(np.sum((orig_ecg_signal - gene_ecg_signal)**2)/np.sum((orig_ecg_signal)**2))*100 
        rmse_ecg = np.sqrt(np.sum((orig_ecg_signal- gene_ecg_signal)**2)/len(orig_ecg_signal))
        rrmse_ecg = np.linalg.norm(orig_ecg_signal- gene_ecg_signal)/np.linalg.norm(orig_ecg_signal)

        prd_ecg_dic[idx] = prd_ecg
        rmse_ecg_dic[idx] = rmse_ecg
        rrmse_ecg_dic[idx] = rrmse_ecg
        
        np.save(os.path.join(sig_path, f'generated_ecg{idx}.npy'), gene_ecg_signal)  
        np.save(os.path.join(sig_path, f'reconstructed_ecg2ecg{idx}.npy'), recon_ecg2ecg_signal)               
        np.save(os.path.join(sig_path, f'reconstructed_ppg2ppg{idx}.npy'), recon_ppg2ppg_signal)  
        np.save(os.path.join(sig_path, f'original_ecg{idx}.npy'), orig_ecg_signal)                
        np.save(os.path.join(sig_path, f'original_ppg{idx}.npy'), orig_ppg_signal)  
 
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines(f"When epoch = {epoch}\n")
        f.write('--'*90 +'\n')
        for idx in ids:
            f.writelines(f"Signal{idx}:\n ECG rmse = {rmse_ecg_dic[idx]:.2f}\n")
            f.writelines(f"ECG rrmse = {rrmse_ecg_dic[idx]:.2f}\n ECG PRD = {prd_ecg_dic[idx]:.2f}\n")
            
    return results_f


class Trainer:
    def __init__(
        self,
        device,
        train_loader,
        test_sample,
        args,
        generator_ppg,
        generator_ecg,
        discriminator_ecg_time,
        discriminator_ecg_freq,
        discriminator_ppg_time,
        discriminator_ppg_freq,
        gene_loss
    ):
        self.device = device
        self.netGE = generator_ecg.to(self.device)
        self.netGP = generator_ppg.to(self.device)
        self.netDEt = discriminator_ecg_time.to(self.device)
        self.netDEf = discriminator_ecg_freq.to(self.device)
        self.netDPt = discriminator_ppg_time.to(self.device)
        self.netDPf = discriminator_ppg_freq.to(self.device)
        self.chk_path = args.chk_pth
        self.optimizerGE =  Adam(self.netGE.parameters(), lr=0.0001)
        self.optimizerGP = Adam(self.netGP.parameters(), lr=0.0001)
        self.optimizerDEt = Adam(self.netDEt.parameters(), lr=0.0001)
        self.optimizerDEf = Adam(self.netDEf.parameters(), lr=0.0001)
        self.optimizerDPt = Adam(self.netDPt.parameters(), lr=0.0001)
        self.optimizerDPf = Adam(self.netDPf.parameters(), lr=0.0001)
        
                
        self.gene_loss = gene_loss
        self.criterion = nn.BCELoss()
        
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.train_loader = train_loader
        self.test_sample = test_sample

        self.ge_errors = []
        self.gp_errors = []
        self.test_ge_errors = []
        self.test_gp_errors = []
        
        self.alpha = 3
        self.beta = 1
        self.gamma = 5
        self.lamb = 30
        
        self.PRD_ecg = []
        self.PRD_ppg = []
        self.rmse_ecg = []
        self.rmse_ppg = []
        self.t= nn.Parameter(torch.tensor([0.07])).to(self.device) 
        self.mse = nn.MSELoss()
        self.qrsloss = QRSLoss(beta=0.5)
  
        

    def clip_loss(self, ppg_zs, ecg_zs, ppg_ws, ecg_ws):
        losses = []
        for i in range(-1,-3,-1):
            ppg_z = ppg_zs[i]
            ecg_z = ecg_zs[i]
            ppg_w = ppg_ws[i]
            ecg_w = ecg_ws[i]

            l = ppg_z.shape[1]
            d = ppg_z.shape[-1]
            ppg_z = ppg_z.squeeze(-2).view((-1, l*d))
            ecg_z = ecg_z.squeeze(-2).view((-1, l*d))

            # Step 2: Multimodal Embedding
            ppg_e = F.normalize(torch.matmul(ppg_z, ppg_w), dim=1)
            ecg_e = F.normalize(torch.matmul(ecg_z, ecg_w), dim=1)
            
            # Step 3: Pairwise Cosine Similarities
            logits = torch.matmul(ppg_e, ecg_e.t()) * torch.exp(self.t)
 
            # Step 4: Symmetric Loss Calculation
            n = ppg_z.shape[0]
            labels = torch.arange(n).to(self.device)
            loss_i = F.cross_entropy(logits, labels, reduction='mean')
            loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
            loss = (loss_i + loss_t) / 2
            losses.append(loss)
        loss =sum(losses)/len(losses)
     
        return loss


    def Discriminator(self, discrim_time, discrim_freq,  fake, targets, discrim_shapelet = None, target_shape_dist=None, fake_shape_dist=None ):

     ### Update Discriminator_time: maximize log(Dt(x)) + log(1 - Dt(G(z))) ###
        real_label = 1
        fake_label = 0
        batch_size = targets.size(0)
  
        real_labels = torch.full((batch_size,), real_label,
                       dtype=targets.dtype, device=self.device)
        out_real = discrim_time(targets)
        out_real = out_real.view(-1)
        errDt_real = self.criterion(out_real, real_labels)

        fake_labels = torch.full((batch_size,), fake_label,
                       dtype=targets.dtype, device=self.device)
        out_false = discrim_time(fake.detach())
        out_false = out_false.view(-1)
        errDt_fake = self.criterion(out_false, fake_labels)

        errDtime = errDt_real + errDt_fake 
     
         ### Update Discriminator_freq: maximize log(Df(x)) + log(1 - D(Gf(z))) ###
        out_real = discrim_freq(targets)
        out_real = out_real.view(-1)
        errDf_real = self.criterion(out_real, real_labels)

        out_false = discrim_freq(fake.detach())
        out_false = out_false.view(-1)
        errDf_fake = self.criterion(out_false, fake_labels)

        errDfreq = errDf_real + errDf_fake 
                
  
         ##### Update Generator: maximaze log(Dt(G(z)))  
        output = discrim_time(fake)
        output = output.view(-1)
        errGtime =  self.criterion(output, real_labels) 
        
        ##### Update Generator: maximaze log(Df(G(z)))  
        output = discrim_freq(fake)
        output = output.view(-1)
        errGfreq =  self.criterion(output, real_labels) 

        return errDtime, errDfreq, errGtime, errGfreq
        
    def _one_epoch(self, epoch):
        all_errGP = 0
        all_errGE = 0
        total_num = 0
        for i, batch in enumerate(self.train_loader, 0):
            real_ppg, real_ecg = batch['ppg'].to(self.device), batch['ecg'].to(self.device)
            exp_rpeaks = batch['exp_rpeaks'].to(self.device)
            total_num += len(real_ppg)

            ppg_recon, ppg_z, ppg_w = self.netGP(real_ppg)
            errGP = self.gene_loss(ppg_recon.squeeze(), real_ppg)
            all_errGP += errGP.item()
            
            ecg_recon, ecg_z, ecg_w   = self.netGE(real_ecg)
            if self.qrsloss is not None:
                errGE = self.qrsloss(ecg_recon.squeeze(), real_ecg, exp_rpeaks)
            else:
                errGE = self.gene_loss(ecg_recon.squeeze(), real_ecg) 
            
            all_errGE += errGE.item()
            
            
            connect_ppg, ppg_w = self.netGP.encode(real_ppg.to(device))
            gen_ecg, attention_maps  = self.netGE.generate(connect_ppg)
            errDPEtime, errDPEfreq, errGPEtime, errGPEfreq  = \
                self.Discriminator(self.netDEt, self.netDEf,  gen_ecg, real_ecg)
            
            if self.qrsloss is not None:
                errGPE = self.qrsloss(gen_ecg.squeeze(), real_ecg, exp_rpeaks) 
            else:
                errGPE = self.gene_loss(gen_ecg.squeeze(), real_ecg) 
            all_errGP += errGPE.item()
  
            clip_loss =  self.clip_loss(connect_ppg, ecg_z, ppg_w, ecg_w)
           
            self.netGP.zero_grad()
            self.netGE.zero_grad()
            self.netDEt.zero_grad()
            self.netDEf.zero_grad()
           
            loss = self.lamb*(errGP + errGE + errGPE + clip_loss)+\
                self.alpha*(errDPEtime + errGPEtime ) +  \
                    self.beta*( errDPEfreq + errGPEfreq ) 


            loss.backward()
        
            self.optimizerGE.step()
            self.optimizerGP.step()
            self.optimizerDEt.step()
            self.optimizerDEf.step()
        

        return  all_errGP/total_num, all_errGE/total_num, batch
        
    def run(self, test_loader):
        for epoch in range(self.num_epochs):
            errGP_, errGE_, batch = self._one_epoch(epoch)
            real_ppg, real_ecg, label = batch['ppg'], batch['ecg'], batch['label']

            self.gp_errors.append(errGP_)
            self.ge_errors.append(errGE_)

            print(f"Epoch: {epoch} | Loss_Generator_PPG: {errGP_} | Loss_Generator_ECG: {errGE_} | Time: {time.strftime('%H:%M:%S')}")

            if epoch % 10 == 0:
                ppg_recon, ppg_z, _ = self.netGP(real_ppg.to(self.device))
                ecg_gen, _  = self.netGE.generate(ppg_z)
                ecg_recon, _, _  = self.netGE(real_ecg.to(self.device))
               
            
                fig, axs = plt.subplots(3, 1, figsize=(15, 12))
                axs[0].plot(real_ppg[0], color='blue', label = 'Real')
                axs[0].plot(ppg_recon.detach().cpu().squeeze(1).numpy()[0], color='red', label = 'Generated')
                axs[0].legend()
                axs[0].set_title("PPG2PPG", fontsize= 12, weight="bold") 
                axs[1].plot(real_ecg[0], color='blue', label = 'Real')
                axs[1].plot(ecg_recon.detach().cpu().squeeze(1).numpy()[0], color='red', label = 'Generated')
                axs[1].legend()
                axs[1].set_title("ECG2ECG", fontsize= 12, weight="bold")             
                axs[2].plot(real_ecg[0], color='blue', label='Real')
                axs[2].plot(ecg_gen.detach().cpu().squeeze(1).numpy()[0], color='red', label='Generated')
                axs[2].legend()
                axs[2].set_title("PPG2ECG", fontsize= 12, weight="bold")
                fig.suptitle(f"training sample  ({label[0]} epoch{epoch})")
                plt.tight_layout()
                plt.show()
                plt.close()
                       
                ppg_recon, ppg_z, _  = self.netGP(self.test_sample['ppg'].to(self.device))
                ecg_gen, _ = self.netGE.generate(ppg_z)
                ecg_recon, _, _  = self.netGE(self.test_sample['ecg'].to(self.device))
                fig, axs = plt.subplots(3, 1, figsize=(15, 12))
                axs[0].plot(self.test_sample['ppg'][0], color='blue', label = 'Real')
                axs[0].plot(ppg_recon.detach().cpu().squeeze(1).numpy()[0], color='red', label = 'Generated')
                axs[0].legend()
                axs[0].set_title("PPG2PPG", fontsize= 12, weight="bold") 
                axs[1].plot(self.test_sample['ecg'][0], color='blue', label = 'Real')
                axs[1].plot(ecg_recon.detach().cpu().squeeze(1).numpy()[0], color='red', label = 'Generated')
                axs[1].legend()
                axs[1].set_title("ECG2ECG", fontsize= 12, weight="bold")             
                axs[2].plot(self.test_sample['ecg'][0], color='blue', label='Real')
                axs[2].plot(ecg_gen.detach().cpu().squeeze(1).numpy()[0], color='red', label='Generated')
                axs[2].legend()
                axs[2].set_title("PPG2ECG", fontsize= 12, weight="bold")
                fig.suptitle(f"testing sample  ({self.test_sample['label'][0]} epoch{epoch})")
                plt.tight_layout()
                plt.show()
                plt.close()
               
            if (epoch+1) % 10 == 0:
                torch.save(self.netGE.state_dict(), self.chk_path+f"generator_ecg{epoch}.pth")
                torch.save(self.netGP.state_dict(), self.chk_path+f"generator_ppg{epoch}.pth")
                torch.save(self.netDEt.state_dict(), self.chk_path+f"Discriminator_ECG_time{epoch}.pth")
                torch.save(self.netDEf.state_dict(), self.chk_path+f"Discriminator_ECG_freq{epoch}.pth")

                
            with torch.no_grad():
                total_num = 0
                all_test_errGP = 0
                all_test_errGE = 0
                for i, batch in enumerate(test_loader, 0):
                    real_ppg = batch['ppg'].to(self.device)
                    real_ecg = batch['ecg'].to(self.device)
                    exp_rpeaks = batch['exp_rpeaks'].to(self.device)
        
                    connect_ppg, ppg_w = self.netGP.encode(real_ppg)
                    generate_ecg, _  = self.netGE.generate(connect_ppg)
                    

                    if self.qrsloss is not None:
                        test_errGE = self.qrsloss(generate_ecg.squeeze(), real_ecg, exp_rpeaks)
                    else:                 
                        test_errGE = self.gene_loss(generate_ecg.squeeze(), real_ecg)
                        
                    all_test_errGE += test_errGE.item()
                    total_num += len(real_ppg)
                    
            self.test_ge_errors.append(all_test_errGE/total_num)
            self.test_gp_errors.append(all_test_errGP/total_num)
      

if __name__=="__main__":  
    
    ids1 = ['07','08','09','16','22','30','34','37','42','43','50','51']
    ids2 = ['01','02','05','11','18','19','20','21','24','29','46']
    ids3 = ['12','14','17','27','35','38','40','47','48']
    ids = ids1 + ids2 + ids3
    
    seg_len = 128
    test_ids = ['07','16','22','42']

    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='')
    parser.add_argument('--test', type=bool, default=True,
                        help='')      
    parser.add_argument('--results', type=str, default= f'./results/bidmc/ClEP_GAN/unseen_{test_ids}_emd/', 
                        help='')  
    parser.add_argument('--chk_pth', type=str, default= f'./results/bidmc/ClEP_GAN/unseen_{test_ids}_emd/CheckPoint/', 
                        help='')  


    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results )
        
    if not os.path.exists(args.chk_pth):
        os.makedirs(args.chk_pth)

    X_train = []
    Y_train = []
    Train_labels = []
    X_test = []
    Y_test = []
    Test_labels = []
    

    for i in range(len(ids)): 
        if ids[i] in test_ids:
            test_ppg_path = f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/real_ppg.npy'
            test_ecg_path =  f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/real_ecg.npy'
            test_label_path =  f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/labels_real.npy'
            times_path =  f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/times.npy'
            
            X_ = np.load(test_ppg_path, allow_pickle=True)
            Y_ = np.load(test_ecg_path, allow_pickle=True)
            labels_ = np.load(test_label_path, allow_pickle=True)
            
            X_test.extend(X_)
            Y_test.extend(Y_)
            Test_labels.extend(labels_)
    
            times = np.load(times_path,  allow_pickle=True)
        else:
            real_ppg_path = f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/real_ppg.npy'
            real_ecg_path =  f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/real_ecg.npy'
            real_label_path =  f'./data/bidmc/processed/{ids[i]}/seg{seg_len}/labels_real.npy'
           
            X_real = np.load(real_ppg_path, allow_pickle=True)
            Y_real = np.load(real_ecg_path, allow_pickle=True)
            real_labels = np.load(real_label_path, allow_pickle=True)    
            
            X_train.extend(X_real)
            Y_train.extend(Y_real)
            Train_labels.extend(real_labels)
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_labels = []
    test_labels = []
 

    for i in range(len(X_train)):
        n_tr = int(len(X_train[i])/2)*2    
        x_overlap, y_overlap = overlap(X_train[i][:n_tr], Y_train[i][:n_tr])
        x_train.extend(x_overlap)
        y_train.extend(y_overlap)
        train_labels.extend(Train_labels[i][:int(n_tr/2)-1])
    for i in range(len(X_test)):  
        n_te = int(len(X_test[i])/2)*2  
        x_overlap, y_overlap = overlap(X_test[i][:n_te], Y_test[i][:n_te])
        x_test.extend(x_overlap)
        y_test.extend(y_overlap)
        test_labels.extend(Test_labels[i][:int(n_te/2)-1])
        
    
    x_train = np.asarray(x_train) 
    y_train = np.asarray(y_train)
    train_labels = np.asarray(train_labels)
    x_test = np.asarray(x_test) 
    y_test = np.asarray(y_test)
    test_labels = np.asarray(test_labels)


    test_ids_dic = {}
    for t in range(len(X_test)):
        key = Test_labels[t][0]
        test_ids_dic[key] = int(n_te/2)-1
        

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float() 
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    
        
        
    train_dataset = ECGDataset(x_train, y_train, train_labels)
    test_dataset = ECGDataset(x_test, y_test, test_labels)

    train_sampler = RandomSampler(train_dataset)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    
    test_sample = next(iter(test_loader))

 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
  
    GP = generator_atten_unet(device, input_channel=1, filter_size=[64, 128, 256, 512, 512, 512],\
                                kernel_size=[31, 31, 31, 31, 31, 31], norm=False)
        
    GE = generator_atten_unet(device, input_channel=1, filter_size=[64, 128, 256, 512, 512, 512],\
                          kernel_size=[31, 31, 31, 31, 31, 31], norm=False) 
        
    DEt = TimeDomainDiscriminator()        
    DEf = FrequencyDomainDiscriminator()
    DPt = TimeDomainDiscriminator()
    DPf = FrequencyDomainDiscriminator()


    trainer = Trainer(device,
                      train_loader, 
                      test_sample,
                      args,
                      generator_ppg=GP,
                      generator_ecg=GE,
                      discriminator_ecg_time = DEt,
                      discriminator_ecg_freq= DEf,
                      discriminator_ppg_time = DPt,
                      discriminator_ppg_freq= DPf,
                      gene_loss= smoothl1_loss) 
    
    trainer.run(test_loader)

    
    test(args, trainer, test_loader, test_ids_dic, args.num_epochs-1)
    
  
   
   