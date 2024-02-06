import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.graphgym.config import cfg

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
        
class Wave_generator(nn.Module):
    def __init__(self, thre=5., hidden_dim=128, nheads=4, n_coe=50, n_scales=4, trans_dropout=0.1, adj_dropout=0.1):
        super().__init__()        
        self.n_coe = n_coe
        self.n_scales = n_scales
        
        self.thre = thre
        
        self.decoder_scaling = nn.Linear(hidden_dim, n_coe)
        self.decoder_wavelet = nn.Linear(hidden_dim, n_coe)
        self.decoder_scales = nn.Linear(hidden_dim, n_scales)
                
        self.ee1 = EigenEncoding(hidden_dim, trans_dropout, nheads)
        #self.ee2 = EigenEncoding(hidden_dim, trans_dropout, nheads)
        

        self.adj_dropout = nn.Dropout(adj_dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def gen_base(self, y, n_coe, flag='scaling'):
        t_even = torch.ones(y.shape).to(y.device)
        t_odd = y
        
        base_wavelet = [t_even.unsqueeze(1)]
        base_scaling = [t_odd.unsqueeze(1)]
        for _ in range(n_coe-1):
            t_even = 2*y*t_odd - t_even
            t_odd = 2*y*t_even - t_odd
            base_wavelet.append(t_even.unsqueeze(1))
            base_scaling.append(t_odd.unsqueeze(1))
        base_wavelet = torch.cat(base_wavelet, 1)
        base_scaling = torch.cat(base_scaling, 1)
        if flag == 'scaling':
            return base_scaling  # B*n_coe*N*1
        elif flag == 'wavelet':
            return base_wavelet  # B*n_coe*N*n_scales
        
    def length_to_mask(self, length):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(length)
        N = length.max().item()
        mask1d  = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)

        mask2d = torch.zeros(B, N, N, device=length.device)
        for i in range(B):
            mask2d[i, :length[i], :length[i]] = 1.0

        # mask1d for key_padding_mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention
        # mask2d for edge selection from padding
        return mask1d, mask2d.bool()
    
    
    def forward(self, batch):    
        max_node = batch.eigenvector.shape[-1]
        max_length = max(batch.length)
        batch_num = batch.eigenvector.shape[0]//max_node
        
        evc_dense = batch.eigenvector.view(batch_num, max_node, max_node)[:, :max_length, :max_length]  # B*N*N
        evc_dense_t = evc_dense.transpose(1, 2)
        eve_dense = batch.eigenvalue.view(batch_num, max_node)[:, :max_length]  # B*N    
        
        eig_mask, edge_idx = self.length_to_mask(batch.length)  # eig_mask B*N  edge_idx B*N*N
        
        eig_filter = self.ee1(eve_dense, eig_mask)
        eig_scales = eig_filter
        
        coe_scaling = self.decoder_scaling(eig_filter)  # B*N*n_coe
        coe_scaling[eig_mask] = 0.
        coe_scaling = self.sigmoid(coe_scaling.sum(1) / (batch.length.view(-1,1)+1e-8))  # B*n_coe
        coe_scaling = coe_scaling / (coe_scaling.sum(-1, keepdim=True)+1e-8)
        
        coe_wavelet = self.decoder_wavelet(eig_filter)  # B*N*n_coe
        coe_wavelet[eig_mask] = 0.
        coe_wavelet = self.sigmoid(coe_wavelet.sum(1) / (batch.length.view(-1,1)+1e-8))  # B*n_coe
        coe_wavelet = coe_wavelet / (coe_wavelet.sum(-1, keepdim=True)+1e-8)
        
        coe_scales = self.decoder_scales(eig_scales)  # B*N*n_scales
        coe_scales[eig_mask] = 0.
        coe_scales = coe_scales.sum(1) / (batch.length.view(-1,1)+1e-8)  # B*n_scales
        batch.coe_scales = coe_scales
        coe_scales = self.sigmoid(coe_scales)*self.thre  # B*n_scales
        
        base_scaling = self.gen_base(eve_dense.unsqueeze(-1)-1., self.n_coe, 'scaling')  # B*n_coe*N*1
        filter_signals_wave = eve_dense.unsqueeze(-1) * coe_scales.unsqueeze(1)
        filter_signals_wave[filter_signals_wave>2.] = 0.
        base_wavelet = self.gen_base(filter_signals_wave-1., self.n_coe, 'wavelet')  # B*n_coe*N*n_scales
        
        base_scaling = 0.5*(-base_scaling+1)
        base_wavelet = 0.5*(-base_wavelet+1)
        
        curr_scaling = coe_scaling.view(batch_num, self.n_coe, 1, 1) * base_scaling  # B*n_coe*N*1
        curr_scaling = curr_scaling.sum(1)  # B*N*1
        
        curr_wavelet = coe_wavelet.view(batch_num, self.n_coe, 1, 1) * base_wavelet  # B*n_coe*N*n_scales
        curr_wavelet = curr_wavelet.sum(1)  # B*N*n_scales
        
        filter_signals_after = torch.cat([curr_scaling, curr_wavelet], -1)  # B*N*(n_scales+1)
        if cfg.lam.tight_use:
            ### tight frame
            filter_signals_after = filter_signals_after / (filter_signals_after.norm(dim=-1, keepdim=True) + 1e-8)  # B*layers*N*(n_scales+1)
        diag_filter_signals_after = torch.diag_embed(filter_signals_after.transpose(2, 1))  # B*(n_scales+1)*N*N
        bases = []
        for i in range(self.n_scales+1):
            filters = evc_dense @ diag_filter_signals_after[:, i, :, :] @ evc_dense_t
            filters = F.normalize(self.adj_dropout(filters))
            bases.append(filters[edge_idx])
        batch.bases = bases
        return batch
        