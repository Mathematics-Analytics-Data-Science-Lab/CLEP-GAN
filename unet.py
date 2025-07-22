import torch
import torch.nn as nn
import torch.nn.functional 

class TimeDomainDiscriminator(nn.Module):
    def __init__(self):
        super(TimeDomainDiscriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, out_shape=None, normalize=True):
            layers = [nn.Conv1d(in_filters, out_filters, 16, stride=2, padding=7)]
            if normalize:
                layers.append(nn.LayerNorm(out_shape))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),  # first layer no normalization
            *discriminator_block(64, 128, out_shape=(128, 128)),
            *discriminator_block(128, 256, out_shape=(256, 64)),
            *discriminator_block(256, 512,  out_shape=(512, 32)),
            nn.Conv1d(512, 1, 16, stride=2, dilation=2)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.model(x)
        out = out.squeeze()
        out = torch.sigmoid(out)
        return out
 
class FrequencyDomainDiscriminator(nn.Module):
    def __init__(self):
        super(FrequencyDomainDiscriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, out_shape=None, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 7, stride=2, padding=3)]
            if normalize:
                layers.append(nn.LayerNorm([out_filters, out_shape[0], out_shape[1]]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),  # first layer no normalization
            *discriminator_block(64, 128, out_shape=(32, 33)),
            *discriminator_block(128, 256, out_shape=(16, 17)),
            *discriminator_block(256, 512, out_shape=(8, 9)),
            nn.Conv2d(512, 1, 8, stride=2, padding=0)
        )

    def forward(self, x):
        window_size = 254
        hop_length = 4
        window = torch.hann_window(window_size).to(x.device)
        stft = torch.stft(x.squeeze(),  n_fft=window_size, hop_length=hop_length, window=window, return_complex=True)
        theta = 1e-10 
        x = torch.log(abs(stft)+theta)
        x = x.unsqueeze(1) 
        out = self.model(x)
        out = out.squeeze()
        out = torch.sigmoid(out)
        return out
    
def Conv1D_(in_channels, out_channel, ksize, strides=1, padding=0, activation=None, use_bias=True):
    op = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=(1, ksize), stride=(1, strides), padding=(0,padding))
    return op

def DeConv1D(in_channel, out_channel, ksize, strides=1, padding=0, use_bias=True):
    op = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, ksize), stride=(1, strides), padding=(0,padding))
    return op

class DownSample(nn.Module):
    def __init__(self, device, input_channel, fsize, ksize, norm, stride_size=2):
        super(DownSample, self).__init__()
        self.norm = norm
        self.paddding = (ksize - stride_size) // 2
        self.Conv1D = Conv1D_(input_channel, fsize, ksize, stride_size, padding=self.paddding, activation=None, use_bias=False)
        self.fsize = fsize
        self.BatchNorm2d = nn.BatchNorm2d
        self.activation = nn.LeakyReLU()
        self.device = device
   
    def forward(self,x):
        layers = []
        layers.append(self.Conv1D)
        if self.norm:
            layers.append(self.BatchNorm2d(self.fsize).to(self.device))
        layers.append(self.activation.to(self.device))
        return nn.Sequential(*layers)(x)
    
class Upsample(nn.Module):
    def __init__(self, device, input_channel, fsize, ksize, norm=True, stride_size=2, drop_rate=0.5, apply_dropout=False): 
        super(Upsample, self).__init__()
        self.padding = (ksize - stride_size) // 2 
        self.DeConv1D = DeConv1D(input_channel, fsize, ksize, stride_size, padding=self.padding, use_bias=False)
        self.BatchNorm2d = nn.BatchNorm2d
        self.dropput = nn.Dropout(drop_rate)
        self.fsize = fsize
        self.norm = norm
        self.apply_dropout = apply_dropout
        self.device = device
    def forward(self, x):
        layers = []
        layers.append(self.DeConv1D)
        if self.norm:
            layers.append(self.BatchNorm2d(self.fsize).to(self.device) )
        if self.apply_dropout:
            layers.append(self.dropput)
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)(x)
    
class attention_block_1d(nn.Module):
    def __init__(self, inter_channel, Conv1D_):
        super(attention_block_1d, self).__init__()  
        self.inter_channel = inter_channel
        self.Conv1D1 = Conv1D_(inter_channel, inter_channel, 1, 1)
        self.Conv1D2 = Conv1D_(inter_channel, inter_channel, 1, 1)
        self.Conv1D3 = Conv1D_(inter_channel, 1, 1, 1)
        self.ReLU = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.permute((0,3,1,2))
        x2 = x2.permute((0,3,1,2))
        
        theta_x = self.Conv1D1(x2)
        phi_g = self.Conv1D2(x1)
        f = self.ReLU(theta_x + phi_g)
        psi_f = self.Conv1D3(f)
       
        rate = self.activation(psi_f)
        att_x = x2 * rate
        att_x = att_x.permute((0,2,3,1))
        return att_x, rate
  
class generator_atten_unet(nn.Module):
    def __init__(self, device, input_channel, filter_size, kernel_size, norm=True, n_downsample=6, skip_connection=True):
        super(generator_atten_unet, self).__init__()
        self.device = device
        self.n_downsample = n_downsample
        self.input_channel = input_channel
        self.norm = norm
        self.skip_connection = skip_connection
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.Downsample_k0 = DownSample(self.device, self.input_channel, self.filter_size[0], self.kernel_size[0], norm=False)
        downsamples = []
        for n in range(1, self.n_downsample):  
            downsamples.append(DownSample(self.device, filter_size[n-1], self.filter_size[n], self.kernel_size[n], norm=self.norm))
        self.Downsample_kn =  nn.Sequential(*downsamples).to(self.device)
        
        attentions = []
        attentions.append(attention_block_1d(7, Conv1D_)) 
        attentions.append(attention_block_1d(15, Conv1D_))  
        attentions.append(attention_block_1d(31, Conv1D_))  
        attentions.append(attention_block_1d(63, Conv1D_))  
        attentions.append(attention_block_1d(127, Conv1D_))  
        attentions.append(attention_block_1d(255, Conv1D_))  
        self.attention_block_1d =  nn.Sequential(*attentions)
        
        upsamples = []
        for n in range(1, self.n_downsample):
            upsamples.append(Upsample(self.device, self.filter_size[self.n_downsample-n], \
                                      self.filter_size[self.n_downsample-n-1], self.kernel_size[self.n_downsample-n-1], self.norm).to(self.device))
        self.Upsample_kn = nn.Sequential(*upsamples)
        self.Upsample_k0 = Upsample(self.device, self.filter_size[self.n_downsample-1],\
                                    self.filter_size[self.n_downsample-1], self.kernel_size[self.n_downsample-1], self.norm, stride_size=1)
    
        padding = (kernel_size[0] - 2) // 2 
        self.DeConv1D = DeConv1D(in_channel=filter_size[0], out_channel=1, ksize = kernel_size[0]+1, strides=2, padding=padding)
        
        self.proj0 = nn.Linear(256, 512*7).to(self.device)
        self.proj1 = nn.Linear(256, 512*15).to(self.device)
        self.proj2 = nn.Linear(256, 512*31).to(self.device)
        self.proj3 = nn.Linear(256, 512*63).to(self.device)
           
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(2)  # Add a dimension for height
        
        # downsample
        connections = []
        for k in range(self.n_downsample):
            if k == 0:
                x = self.Downsample_k0(x) 
            else:
                x = self.Downsample_kn[k-1](x)
            connections.append(x)
        x_weight = [self.proj2.weight,self.proj1.weight,self.proj0.weight]

        # upsampling
        x =  self.Upsample_k0(x)
        if self.skip_connection:
            _x, rate = self.attention_block_1d[0](x, connections[self.n_downsample-1])
            x = x + _x
        
        for l in range(1, self.n_downsample):
            x = self.Upsample_kn[l-1](x)
            if self.skip_connection:
                _x, rate = self.attention_block_1d[l](x, connections[k-l])
                x = x + _x

        x = self.DeConv1D(x)
        x = nn.Tanh()(x)
        x = x.squeeze(1)
        out = x.squeeze(1)
        return out, connections, x_weight
    
    def encode(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(2)  
        
        # downsample
        connections = []
        for k in range(self.n_downsample):
            if k == 0:
                x = self.Downsample_k0(x) 
            else:
                x = self.Downsample_kn[k-1](x)
            connections.append(x)
        xz_weight = [self.proj2.weight,self.proj1.weight,self.proj0.weight]
        return connections, xz_weight

    def generate(self, connections):
        # upsampling
        x =  self.Upsample_k0(connections[-1])
        if self.skip_connection:
            _x, rate = self.attention_block_1d[0](x, connections[self.n_downsample-1])
            x = x + _x
        
        for l in range(1, self.n_downsample):
            x = self.Upsample_kn[l-1](x)
            if self.skip_connection:
                k = self.n_downsample - 1
                _x, rate = self.attention_block_1d[l](x, connections[k-l])
                x = x + _x

        x = self.DeConv1D(x)
        x = nn.Tanh()(x)
        x = x.squeeze(1)
        out = x.squeeze(1)
        return out, rate  