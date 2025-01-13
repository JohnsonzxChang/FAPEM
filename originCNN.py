import torch as th
import numpy as np
from torch import nn
import math 
import einops as eth
from einops.layers.torch import Rearrange
from torch.nn import functional as FF
from conf import Config

class mine_identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(mine_identity, self).__init__()

    def forward(self, x):
        return x

def get_norm(norm_str):
    if norm_str == 'batch':
        return nn.BatchNorm2d
    elif norm_str == 'batch2':
        return nn.BatchNorm2d
    elif norm_str == 'batch1':
        return nn.BatchNorm1d
    elif norm_str == 'layer':
        return nn.LayerNorm
    elif norm_str == 'none':
        return mine_identity
    elif norm_str is None:
        return mine_identity
    else:
        raise ValueError(f'norm {norm_str} not Implemented')

def get_non_linear(non_linear_str):
    if non_linear_str == 'relu':
        return nn.ReLU
    elif non_linear_str == 'gelu':
        return nn.GELU
    elif non_linear_str == 'none':
        return mine_identity
    elif non_linear_str is None:
        return mine_identity
    else:
        raise ValueError(f'non_linear {non_linear_str} not Implemented')

def get_conv_norm_non_linear(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, non_linear):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        get_norm(norm)(out_channels),
        get_non_linear(non_linear)()
    )

def get_conv_norm_non_linear_upsample(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, non_linear):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 0, groups, bias, dilation),
        get_norm(norm)(out_channels),
        get_non_linear(non_linear)()
    )

class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # input shape: (b, c, h, w)
        # attention shape: (b, c, 1, 1)
        y = th.mean(x, dim=(2, 3))
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)

class Freq_Block(nn.Module):
    def __init__(self, C, reduction=16, FreqList=None, TN=50, Fs=250, H=3, M=10):
        super(Freq_Block, self).__init__()
        if FreqList is None:
            FreqList = np.linspace(8, 15.8, 40)
        FreqList = th.tensor(FreqList).float()
        assert type(TN) is int
        self.TN = TN
        position = th.arange(self.TN).unsqueeze(1) / Fs
        pe_sin = []
        pe_cos = []
        for i in range(H):
            pe_sin.append(th.sin(math.pi * 2 * position * FreqList * (i+1)).unsqueeze(0))  # 1, T, F
            pe_cos.append(th.cos(math.pi * 2 * position * FreqList * (i+1)).unsqueeze(0))  # 1, T, F
        self.pe_sin = th.cat(pe_sin, dim=0) # H, T, F
        self.pe_sin.requires_grad = False
        self.pe_cos = th.cat(pe_cos, dim=0) # H, T, F
        self.pe_cos.requires_grad = False
        if M == 1:
            self.wx = False
        else:
            self.wx = nn.Parameter(th.randn(M))
        self.wy = nn.Parameter(th.randn(H))
        self.wc = nn.Parameter(th.ones(C)/C)
        self.fc = nn.Sequential(
            nn.Linear(C, C // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // reduction, C, bias=False),
            nn.Sigmoid()
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(C*FreqList.shape[0], C*FreqList.shape[0] // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(C*FreqList.shape[0] // reduction, C*FreqList.shape[0]),
        #     nn.Sigmoid()
        # )
        print(f'pe shape: {self.pe_sin.shape}')
        
    def to(self, device):
        print(f'pe device: {device}')
        self.pe_sin = self.pe_sin.to(device)
        self.pe_cos = self.pe_cos.to(device)
        return super().to(device)

    def cca_x(self, x):
        if self.wx is False:
            assert x.shape[2] == 1
            return x.squeeze(2)
        else:
            return th.einsum('bcmt,m->bct', x, self.wx)
    
    def cca_y(self):
        return [th.einsum('htf,h->ft', self.pe_sin, self.wy), 
                th.einsum('htf,h->ft', self.pe_cos, self.wy)]
        
    def corr(self, x, eps=1e-5):
        # x: {B, C, M, T}; ref: {T, F, H}
        # return: {B, C, F}
        xs = self.cca_x(x)
        [pe_sin_s, pe_cos_s] = self.cca_y()
        assert xs.shape[2] == pe_sin_s.shape[1], f'{xs.shape} & {pe_sin_s.shape}'
        xy = th.einsum('bct,ft->bcf', xs, pe_sin_s
            ) ** 2 + th.einsum('bct,ft->bcf', xs, pe_cos_s
            ) ** 2
        assert th.sum(th.isnan(xy)) == 0, print(xy.shape)
        return xy
    
    def forward(self, x):
        # input shape: (b, c, m, t)
        # attention shape: (b, c, f)
        # print(f'input shape: {x.shape}')
        xy = self.corr(x)
        # att_shape = xy.shape
        y = th.mean(xy, dim=2)
        y = self.fc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x), th.einsum('bcf,c->bf', th.softmax(xy, dim=-1), self.wc).unsqueeze(dim=1)
    # th.mean(xy, dim=1, keepdim=True)
    # th.einsum('bcf,c->bf', xy, self.wc).unsqueeze(dim=1) # b, 1, f
    
    def forward1(self, x):
        # input shape: (b, c, m, t)
        # attention shape: (b, c, f)
        # print(f'input shape: {x.shape}')
        xy = self.corr(x)
        att_shape = xy.shape
        xy = self.fc(xy.reshape(att_shape[0], -1)).reshape(att_shape)
        y = th.mean(xy, dim=2, keepdim=True).unsqueeze(-1)
        return x * y.expand_as(x), xy
        
class G_Net(nn.Module):
    def __init__(self, N, t, T, F, s, p, emb, norm, non_linear, drop, bias, mid, device, ica, M, H, compress, chn_att=True):
        super().__init__()
        self.L = 40
        self.drop = drop
        assert len(N)-1 == len(t) == len(s) and N[0] == 3
        FS, TS = [], []
        tmp = T
        tmpF = F
        for i in range(len(N)-1):
            TS.append(tmp)
            FS.append(tmpF)
            tmpF = tmpF / tmp * (1 + int(math.floor((tmp - t[i] + 2*p[i]) / s[i])))
            tmp = 1 + int(math.floor((tmp - t[i] + 2*p[i]) / s[i]))
        TS.append(tmp)
        FS.append(tmpF)
        
        # if params['dataset'] == 'bench':
        self.num_person = 35  
        # elif params['dataset'] == 'beta':
        #     self.num_person = 70
        # else:
        #     raise ValueError('dataset not implemented')
        self.H = H #params['H']
        self.device = device
        self.compress = compress
        # self.head = params['head']  # 6

        self.id_embedding = nn.Embedding(self.num_person, emb, max_norm=1)

        self.id_operation = nn.Sequential(
            nn.Linear(emb, (M + 1)* ica, bias=False),
            # nn.Dropout(drop[0]),
            Rearrange('b (m n) -> b m n', m=(M+1)),
        )

        self.preprocess = nn.Sequential(
            get_conv_norm_non_linear(N[0], N[1], (1,t[0]), (1,s[0]), (0,p[0]), (1,1), 1, bias[0], norm, non_linear),
            nn.Dropout(self.drop[0]),
        )

        self.chn_combination = nn.Sequential(
            get_conv_norm_non_linear(N[1], N[2], (ica,t[1]), (1,s[1]), (0,p[1]), (1,1), 1, bias[1], norm, non_linear),
            nn.Dropout(self.drop[1]),
        )
        if chn_att:
            self.chn_attention = nn.Sequential(
                Freq_Block(N[2], self.compress, None, TS[2], FS[2], self.H[0], 1).to(self.device),
            )
        else:
            self.chn_attention = nn.Identity()

        self.tempo_fea, self.tempo_down, self.tempo_att = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(2, len(t)):
            tmp = self._create_tempo_layers(t[i], s[i], p[i], N[i], N[i+1], TS[i+1], FS[i], bias, norm, non_linear,
                                            har=self.H[i-1])
            self.tempo_fea.append(tmp[0])
            self.tempo_down.append(tmp[1])
            self.tempo_att.append(tmp[2])

        self.flat = N[-1] * TS[-1] # + 40 * (len(t)-1)
        self.function_resnet = get_non_linear(non_linear)()
        self.fc_linear = nn.Sequential(
            nn.Dropout(self.drop[-1]),
            # get_non_linear(non_linear)(),
            nn.Linear(self.flat, mid, bias=bias[-1]),
        )
        self.att_linear = nn.Sequential(
            nn.Dropout(self.drop[1]),
            get_non_linear(non_linear)(),
            nn.Linear(40 * (len(t)-1), mid, bias=True),
            
        )
        print('--------------------------------------------------')
        print('--------------------------------------------------')
        print(f'final flatten number is {self.flat}...')
        print(f'K, P, S, har ...{t},,{p},,{s},,{self.H}')
        print(f'N, TS, FS ...{N},,{TS},,{FS}')
        print(f'emb, ica, M ...{emb},,{ica},,{M}')
        print(f'norm, non_linear ...{norm},,{non_linear}')
        print(f'bias, drop ...{bias},,{self.drop}')
        print(f'flatten dim...{self.flat}')
        print('--------------------------------------------------')
        print('--------------------------------------------------')

    def _create_tempo_layers(self, K, S, P, Cin, Cout, TN, Fs, bias, norm, non_linear, har):
        feature = nn.Sequential(
            get_conv_norm_non_linear(Cin, Cout, (1,K), (1,S), (0,P), (1,1), 1, bias[2], norm, non_linear),
            nn.Dropout(self.drop[2])
        )
        downsample = nn.Sequential(
            nn.Conv2d(Cin, Cout, 1, (1, S), bias=False),
            get_norm(norm)(Cout),
        ) if S != 1 else nn.Identity()
        attention = nn.Sequential(
            Freq_Block(Cout, self.compress, None, TN, Fs, har, M=1).to(self.device),
        ) if Fs > 70 else nn.Identity()
        return feature, downsample, attention
    
    def f1x(self, x):
        # print(f'f1x input shape: {x.shape}')
        x = self.preprocess(x)
        # b c m t
        x_m = th.mean(x, dim=2, keepdim=True)
        x = x - x_m.expand_as(x)
        x = th.cat((x, x_m), dim=2)
        # assert x.shape[2] == 10
        return x
    
    def f1id(self, id):
        return self.id_embedding(id)
    
    def f1p(self, idp):
        # print(f'f1p input shape: {idp.shape}')
        return self.id_operation(idp)
    
    def f2(self, x):
        # print(f'f2 input shape: {x.shape}')
        x = self.chn_combination(x)
        x_att = self.chn_attention(x)
        x = self.function_resnet(x + x_att[0])
        x = FF.dropout(x, self.drop[0], training=self.training)
        return x, x_att[1]
    
    def f3(self, x):
        # print(f'f3 input shape: {x.shape}')
        x_att = []
        for i in range(len(self.tempo_fea)):
            xtmp = self.tempo_fea[i](x)
            # print(f'xtmp shape: {xtmp.shape}')
            xtmp, att_tmp = self.tempo_att[i](xtmp)
            # print(f': {xtmp.shape}')
            x_att.append(att_tmp)
            x = self.function_resnet(xtmp + self.tempo_down[i](x))
            # print(f'f3-loop input shape: {x.shape}')
            if not i == len(self.tempo_fea) - 1:
                x = FF.dropout(x, self.drop[0], training=self.training)
        return x, x_att
    
    def forward(self, x, id=None):
        att = []
        if id is None:
            id = th.randint(0, self.num_person, (x.shape[0],)).to(self.device)
        id = self.f1id(id)
        x = self.f1x(x)
        x_p = th.einsum('bcmt,bmn->bcnt', x, self.f1p(id))
        
        x, x_att = self.f2(x_p)
        att.append(x_att)
        x, x_att = self.f3(x)
        att.extend(x_att)
        att = th.cat(att, dim=1) # b 4 f
        xot = eth.rearrange(x, 'b c 1 t -> b (c t)')
        att = eth.rearrange(att, 'b c f -> b (c f)')
        att = self.att_linear(att)
        x = self.fc_linear(xot)
        # print(f'att shape: {att.shape}; x shape: {x.shape}')
        return th.cat((x, att), dim=1), att, xot
    
class C_Net(nn.Module):
    def __init__(self, N, t, T, F, s, p, emb, norm, non_linear, drop, bias, mid, device, ica, M, H, compress, chn_att=True):
        super().__init__()
        self.L = 40
        FS, TS = [], []
        tmp = T
        tmpF = F
        for i in range(len(N)-1):
            TS.append(tmp)
            FS.append(tmpF)
            tmpF = tmpF / tmp * (1 + int(math.floor((tmp - t[i] + 2*p[i]) / s[i])))
            tmp = 1 + int(math.floor((tmp - t[i] + 2*p[i]) / s[i]))
        TS.append(tmp)
        self.flat = N[-1] * TS[-1] + 40 * (len(t)-1)
        self.fc = nn.Sequential(
            get_non_linear(non_linear)(),
            nn.Dropout(drop[1]),
            nn.Linear(mid*2, self.L, bias=bias[-1]),
        )

    def forward(self, z):
        x = self.fc(z)
        return x

class SSVEP_Net(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        assert isinstance(params, Config)
        self.G = G_Net(params.N, params.t, params.seq_len, params.F, params.s, params.p, params.emb, params.norm, params.non_linear, params.drop, params.bias, params.mid, params.device, params.ica, params.enc_in, params.H, params.compress, params.chn_att)
        self.C = C_Net(params.N, params.t, params.seq_len, params.F, params.s, params.p, params.emb, params.norm, params.non_linear, params.drop, params.bias, params.mid, params.device, params.ica, params.enc_in, params.H, params.compress, params.chn_att)

    def forward(self, x, id=None):
        z, att, xot = self.G(x, id)
        y = self.C(z)
        return y
    
if __name__ == '__main__':
    model = SSVEP_Net(params=Config())
    from torchinfo import summary
    summary(model, input_size=(2, 3, 9, 50))