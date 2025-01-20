import torch
import torch.nn as nn
import torch.nn.functional as FF
from conf import Config
import numpy as np

ALL_NONLINEAR = {'relu':nn.ReLU, 'relu6':nn.ReLU6, 'elu':nn.ELU, 'leaky':nn.LeakyReLU, 'gelu':nn.GELU, 'selu':nn.SELU, 'none':None}
ALL_NORMALIZATION = {'batch':nn.BatchNorm2d, 'none':None}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, M, act, norm, drop, down, T=50):
        super().__init__()
        assert act in ALL_NONLINEAR.keys(), f'act must be in {ALL_NONLINEAR.keys()}'
        assert norm in ALL_NORMALIZATION.keys(), f'norm must be in {ALL_NORMALIZATION.keys()}'
        self.act = ALL_NONLINEAR[act]() if act != 'none' else nn.Identity()
        self.norm = ALL_NORMALIZATION[norm](out_channels) if norm != 'none' else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, padding='same', bias=False),
            # self.norm,
            # self.act,
            # nn.Dropout(drop)
        )
        if down:
            self.feature = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=(1,k), stride=(1,s), padding=(0,p), bias=False) 
                for _ in range(M)])
        else:
            self.feature = nn.ModuleList([ 
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(1,k), stride=(1,s), padding=(0,p), bias=False, output_padding=(0,1) if (T % s == 0 and s != 1) else 0) 
                for _ in range(M)])
        
    def forward(self, x):
        # b c1 m t1 -> b c2 m t2
        x = self.adapt(x)
        res = []
        for i in range(len(self.feature)):
            tmp = x[:,:,i,:].unsqueeze(2)
            # print(tmp.shape)
            res.append(self.feature[i](tmp))
        res = torch.cat(res, dim=2)
        # print(res.shape)
        x = self.norm(res)
        x = self.act(x)
        x = self.drop(x)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        assert isinstance(conf, Config)
        num_classes= conf.num_class # 40
        M = conf.enc_in # 10
        C0 = conf.N[0] # 3
        drop = conf.drop[2:] # [0.1, 0.9]
        self.num_classes = num_classes
        C = [C0, 32, 32, 64]
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SELU(),
            nn.Linear(32, 32)
        )
        
        # 条件类别嵌入
        self.class_embed = nn.Embedding(num_classes, 32)
        
        self.cond_proj = nn.Sequential(
            nn.Linear(32, C[-1]),  # 将128维扩展到256维匹配特征图
            nn.SELU()
        )
        
        # 下采样路径
        self.down = nn.ModuleList([
            Block(C[0], C[1], k=5, s=1, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=True),
            Block(C[1], C[2], k=5, s=2, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=True),
            Block(C[2], C[3], k=5, s=2, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=True),
        ])
        self.downChn = nn.ModuleList([
            nn.Conv2d(C[1], C[1], (M,1), 1, padding='same', bias=False),
            nn.Conv2d(C[2], C[2], (M,1), 1, padding='same', bias=False),
            nn.Conv2d(C[3], C[3], (M,1), 1, padding='same', bias=False)
        ])
        
        
        
        # 上采样路径
        self.up = nn.ModuleList([
            # 中间层
            Block(C[3], C[3], k=5, s=1, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=False, T=13),
            Block(C[3], C[2], k=5, s=2, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=False, T=25),
            Block(C[2], C[1], k=5, s=2, p=2, M=M, act='selu', norm='batch', drop=drop[0], down=False, T=50),
            
        ])
        self.upChn = nn.ModuleList([
            nn.Conv2d(C[3], C[3], (M,1), 1, padding='same', bias=False),
            nn.Conv2d(C[2], C[2], (M,1), 1, padding='same', bias=False),
            nn.Conv2d(C[1], C[1], (M,1), 1, padding='same', bias=False),
            
        ])
        self.out = nn.Conv2d(C[1], C[0], 1, 1, padding='same', bias=False)
        self.T = conf.seq_len # 10
        self.cla_linear = nn.Linear(80*num_classes, num_classes)
        self.cla_weight = nn.ModuleList([nn.Linear((C0*M), 1) for _ in range(num_classes)
                                         ])
        ref_freq = np.linspace(8, 15.8, 40).reshape(-1,1)
        t_span = np.arange(self.T) / 250
        self.ref_freq = np.concatenate([np.sin(2*np.pi*ref_freq*t_span), np.cos(2*np.pi*ref_freq*t_span)], axis=0)
        assert self.ref_freq.shape == (80, self.T)
        self.ref_freq = torch.tensor(self.ref_freq, dtype=torch.float32, requires_grad=False)
        
    def forward(self, x, t, labels=None):
        # assert x.shape == (2, 3, 9, 50), f'x shape must be (2, 3, 9, 50), but got {x.shape}'
        if labels is None:
            labels = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)
        # 嵌入时间和类别
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        c_emb = self.class_embed(labels)
        
        # 合并条件信息
        cond = self.cond_proj(t_emb + c_emb).unsqueeze(-1).unsqueeze(-1)
        
        # 下采样
        res = []
        for fcn, fcnChn in zip(self.down, self.downChn):
            # print(x.shape)
            x = FF.relu(fcn(x))
            x = x + fcnChn(x)
            # print(x.shape)
            res.append(x)
            
        res[-1] = res[-1] + cond
        x = res[-1]
        # 上采样
        for fcn, fcnChn in zip(self.up, self.upChn):
            x = FF.selu(fcn(x))
            x = x + fcnChn(x)
            # print(x.shape, res[-1].shape)
            x = x + res.pop()   
        
        return self.out(x)
    
    def combine_all(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)
        res = []
        for i in range(self.num_classes):
            res.append(self.cla_weight[i](x).squeeze(-1))
        return torch.stack(res, dim=1)
    
    def combine(self, x):
        res = []
        for i in range(self.num_classes):
            xs = x[:,i,:,:,:].reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)
            res.append(self.cla_weight[i](xs).squeeze(-1))
        return torch.stack(res, dim=1)
            
    def corr(self, x, y):
        # y = self.ref_freq.to(x.device)
        assert x.shape == y.shape, f'x shape must be equal to y shape, but got {x.shape} and {y.shape}'
        y = torch.einsum('bft,bgt->bfg', x, y) / x.shape[-1]
        y = y * torch.eye(y.shape[-1]).to(y.device).unsqueeze(0).expand_as(y)
        y = y.sum(dim=-1)
        # y = y.reshape(y.shape[0], -1)
        return y
            

class DiffusionModel:
    def __init__(self, conf):
        assert isinstance(conf, Config)
        num_steps = conf.diff_num_step # 1000, 
        beta_start = conf.diff_beta[0] # 1e-4, 
        beta_end = conf.diff_beta[1] # 0.02, 
        device = conf.device # 'cuda'
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def forward_diffusion(self, x0, t):
        alpha_t = self.alpha_bar[t]
        noise = torch.randn_like(x0)
        return (
            torch.sqrt(alpha_t)[:, None, None, None] * x0 + 
            torch.sqrt(1 - alpha_t)[:, None, None, None] * noise
        ), noise
        
    @torch.no_grad()
    def sample(self, model, n_samples, labels, device):
        x = torch.randn(n_samples, 1, 28, 28).to(device)
        
        for t in range(self.num_steps-1, -1, -1):
            t_batch = torch.ones(n_samples, dtype=torch.long).to(device) * t
            predicted_noise = model(x, t_batch, labels)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = 1 / torch.sqrt(alpha_t) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
            ) + torch.sqrt(self.beta[t]) * noise
            
        return x
    
    @torch.no_grad()
    def sample_all_class(self, model, all_label, shapes, device):
        n_samples = shapes[0]
        res = [] 
        for lb in all_label:
            labels = torch.ones(n_samples, dtype=torch.long).to(device) * lb
            x = torch.randn(shapes).to(device)
            # t_batch = torch.zeros(n_samples, dtype=torch.long).to(device)
            for t in range(self.num_steps-1, -1, -40):
                t_batch = torch.ones(n_samples, dtype=torch.long).to(device) * t
                predicted_noise = model(x, t_batch, labels)
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                    
                x = 1 / torch.sqrt(alpha_t) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                ) + torch.sqrt(self.beta[t]) * noise
            
            res.append(x)
        res = torch.stack(res, dim=1)
        return res 

def train_diffusion(batch, model, diffusion, optimizer, e, device):
        images = batch['data'].to(device)
        labels = batch['label'].to(device)
        
        # 随机时间步
        t = torch.randint(0, diffusion.num_steps, (images.shape[0],)).to(device)
        
        # 前向扩散
        noisy_images, noise = diffusion.forward_diffusion(images, t)
        
        # 预测噪声
        predicted_noise = model(noisy_images, t, labels)
        
        # 计算损失
        loss = FF.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss

def sample_model(model, diffusion, labelIn, device):
    # 采样
    model.eval()
    with torch.no_grad():
        samples = diffusion.sample(model, labelIn.shape[0], labelIn, device)
    return samples.cpu()

if __name__ == "__main__":
    from torchinfo import summary
    model = ConditionalUNet(Config())
    summary(model.to('cuda:0'), input_size=((2, 3, 9, 50), (2,)), device='cuda:0')
    # model.predict(torch.randn(size=(2, 3, 9, 50)), torch.randint(0, 40, (2,)))
    
    # model = Block(32, 32, k=5, s=2, p=2, M=9, act='relu', norm='batch', drop=0.5, down=True)
    # summary(model, input_size=(2, 32, 9, 50), device='cpu')