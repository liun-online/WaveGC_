import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.graphgym.config import cfg
from einops import rearrange
from graphgps.transform.transforms import path_dict

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [B, N]
        # output: [B, N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)
        
        return self.eig_w(eeig)

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class EigenEncoding(nn.Module):
    def __init__(self, hidden_dim, trans_dropout, nheads):
        super().__init__()  
        self.eig_encoder = SineEncoding(hidden_dim)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(trans_dropout)
        self.ffn_dropout = nn.Dropout(trans_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, trans_dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, eve_dense, eig_mask):
        eig = self.eig_encoder(eve_dense)  # B*N*d
        mha_eig = self.mha_norm(eig)  # B*N*d
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=eig_mask, average_attn_weights=False)
        eig = eig + self.mha_dropout(mha_eig)
        
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)  # B*N*d
        return eig
        
class Wave_generatorV2(nn.Module):
    def __init__(self, layers=4, low_thre=0.5, high_thre=10., hidden_dim=128, nheads=4, n_coe=50, n_scales=4, trans_dropout=0.1):
        super().__init__()        
        self.n_coe = n_coe
        self.n_scales = n_scales
        self.layers = layers
                
        self.decoder_scaling = nn.Linear(hidden_dim, n_coe*layers)
        self.decoder_wavelet = nn.Linear(hidden_dim, n_coe*layers)
        self.decoder_scales = nn.Linear(hidden_dim, n_scales*layers)
                
        self.ee1 = EigenEncoding(hidden_dim, trans_dropout, nheads)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.scale_scale = torch.ones(1, 1, n_scales, dtype=torch.float32).cuda()
        self.scale_scale[:,:,0] *= low_thre
        self.scale_scale[:,:,2:] *= high_thre
        
    def gen_base(self, y, n_coe, flag='scaling'):
        t_even = torch.ones(y.shape).to(y.device)
        t_odd = y
        
        base_wavelet = [t_even.unsqueeze(2)]
        base_scaling = [t_odd.unsqueeze(2)]
        for _ in range(n_coe-1):
            t_even = 2*y*t_odd - t_even
            t_odd = 2*y*t_even - t_odd
            base_wavelet.append(t_even.unsqueeze(2))
            base_scaling.append(t_odd.unsqueeze(2))
        base_wavelet = torch.cat(base_wavelet, 2)
        base_scaling = torch.cat(base_scaling, 2)
        if flag == 'scaling':
            return base_scaling  # B*n_coe*sele_num*1
        elif flag == 'wavelet':
            return base_wavelet  # B*n_coe*sele_num*n_scales
        
    def length_to_mask(self, length, sele_num):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(length)
        N = length.max().item()
        S = sele_num.max().item()
        mask1d  = torch.arange(S, device=length.device).expand(B, S) >= sele_num.unsqueeze(1)

        mask2d = torch.zeros(B, N, N, device=length.device)
        for i in range(B):
            mask2d[i, :length[i], :length[i]] = 1.0

        # mask1d for key_padding_mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention
        # mask2d for edge selection from padding
        return mask1d, mask2d.bool()
    
    
    def forward(self, batch):      
        max_node = max(batch.length)
        max_length = max(batch.length)
        sele_num = max(batch.sele_num)
        batch_num = 1
        
        eva_dense = batch.eigenvalue.view(batch_num, sele_num)  # 1*sele_num 
        
        eig_mask, edge_idx = self.length_to_mask(batch.length, batch.sele_num)  # eig_mask 1*sele_num   edge_idx 1*N*N
        
        eig_filter = self.ee1(eva_dense, eig_mask)  # 1*sele_num*d
        eig_scales = eig_filter
        
        coe_scaling = self.decoder_scaling(eig_filter)  # 1*sele_num*(n_coe*layers)
        coe_scaling[eig_mask] = 0.
        coe_scaling = rearrange(coe_scaling, 'a b (c d) -> a d b c', c=self.n_coe, d=self.layers)  # 1*layers*sele_num*n_coe
        coe_scaling = F.softmax(coe_scaling.sum(2) / (batch.length.view(-1,1,1)+1e-8), dim=-1)  # 1*layers*n_coe
        
        coe_wavelet = self.decoder_wavelet(eig_filter)  # 1*sele_num*(n_coe*layers)
        coe_wavelet[eig_mask] = 0.
        coe_wavelet = rearrange(coe_wavelet, 'a b (c d) -> a d b c', c=self.n_coe, d=self.layers)  # 1*layers*sele_num*n_coe
        coe_wavelet = F.softmax(coe_wavelet.sum(2) / (batch.length.view(-1,1,1)+1e-8), dim=-1)  # 1*layers*n_coe
        
        coe_scales = self.decoder_scales(eig_scales)  # 1*sele_num*(n_scales*layers)
        coe_scales[eig_mask] = 0.
        coe_scales = rearrange(coe_scales, 'a b (c d) -> a d b c', c=self.n_scales, d=self.layers)  # 1*layers*sele_num*n_scales        
        coe_scales = coe_scales.sum(2) / (batch.length.view(-1,1,1)+1e-8)  # 1*layers*n_scales
        batch.coe_scales = coe_scales
        coe_scales = self.sigmoid(coe_scales)*self.scale_scale  # 1*layers*n_scales   
        
        eva_expand = eva_dense.unsqueeze(-1).unsqueeze(1).expand(batch_num, self.layers, sele_num, 1)  # 1*layers*sele_num*1 
        
        base_scaling = self.gen_base(eva_expand-1., self.n_coe, 'scaling')  # 1*layers*n_coe*sele_num*1
        filter_signals_wave = eva_expand * coe_scales.unsqueeze(2)  # 1*layers*sele_num*n_scales
        filter_signals_wave[filter_signals_wave>2.] = 0.
        base_wavelet = self.gen_base(filter_signals_wave-1., self.n_coe, 'wavelet')  # 1*layers*n_coe*sele_num*n_scales
        
        base_scaling = 0.5*(-base_scaling+1)
        base_wavelet = 0.5*(-base_wavelet+1)
        
        curr_scaling = coe_scaling.view(batch_num, self.layers, self.n_coe, 1, 1) * base_scaling  # 1*layers*n_coe*sele_num*1
        curr_scaling = curr_scaling.sum(2)  # 1*layers*sele_num*1
        
        curr_wavelet = coe_wavelet.view(batch_num, self.layers, self.n_coe, 1, 1) * base_wavelet  # 1*layers*n_coe*sele_num*n_scales
        curr_wavelet = curr_wavelet.sum(2)  # 1*layers*sele_num*n_scales
        
        filter_signals_after = torch.cat([curr_scaling, curr_wavelet], -1)  # 1*layers*sele_num*(n_scales+1)
        if cfg.lam.tight_use:
            ### tight frame
            filter_signals_after = filter_signals_after / (filter_signals_after.norm(dim=-1, keepdim=True) + 1e-8)  # 1*layers*sele_num*(n_scales+1)
        
        batch.filter_signals_after = filter_signals_after # 1*layers*sele_num*(n_scales+1)
        batch.edge_idx = edge_idx
        return batch