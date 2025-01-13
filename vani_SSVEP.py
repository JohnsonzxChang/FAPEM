from Patch_SSVEP import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math
from conf import Config

class AttentionInterface(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class SpatialAttention(AttentionInterface):
    def __init__(self, config):
        super().__init__()
        assert isinstance(config, Config)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
    def forward(self, q, k, v, mask=None):
        # bt m d
        batch_size = q.size(0)
        
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return out
    
class TemporalAttention(AttentionInterface):
    def __init__(self, config):
        super().__init__()
        assert isinstance(config, Config)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
    def forward(self, q, k, v, mask=None):
        # bm t d
        batch_size = q.size(0)
        
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, attention_layer: Optional[AttentionInterface] = None):
        super().__init__()
        assert isinstance(config, Config)
        self.m = config.enc_in
        self.t = config.data_t0
        self.d_model = config.d_model
        # 第一阶段
        self.attention1 = attention_layer or TemporalAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # 第二阶段
        self.attention2 = attention_layer or SpatialAttention(config)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm4 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        # b m t d
        x = x.reshape(-1, self.t, self.d_model)
        # 第一阶段
        attn1_out = self.attention1(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn1_out))
        ff1_out = self.ff1(x)
        x = self.norm2(x + self.dropout(ff1_out))
        x = x.reshape(-1, self.m, self.t, self.d_model)
        x = x.permute(0, 2, 1, 3)
        # b t m d
        x = x.reshape(-1, self.m, self.d_model)
        # 第二阶段
        attn2_out = self.attention2(x, x, x, mask)
        x = self.norm3(x + self.dropout(attn2_out))
        ff2_out = self.ff2(x)
        x = self.norm4(x + self.dropout(ff2_out))
        x = x.reshape(-1, self.t, self.m, self.d_model)
        x = x.permute(0, 2, 1, 3)
        return x

class VanillaTransformer(nn.Module):
    def __init__(self, config, 
                 attention_factory = None):
        super().__init__()
        assert isinstance(config, Config)
        self.config = config
        
        # Embedding layers
        self.embedding = nn.Linear(3, config.d_model)
        self.pos_encoding = self._create_positional_encoding()
        print(self.pos_encoding.shape)
        
        # Create encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config,
                attention_factory(config) if attention_factory else None
            ) for _ in range(config.e_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(config.d_model, 3)
        
    def _create_positional_encoding(self):
        max_seq_len = 100
        pe = torch.zeros(max_seq_len, self.config.d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2) * 
                           -(math.log(10000.0) / self.config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0).unsqueeze(0), requires_grad=False)
        
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 3, 1)
        
        x = self.embedding(x)
        print(x.shape)
        x = x + self.pos_encoding[:, :, :x.size(2), :]
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
            
        return self.output_layer(x)
    
    
if __name__ == '__main__':
    config = Config()
    model = VanillaTransformer(config)
    from torchinfo import summary
    summary(model, input_size=(2, 3, 9, 50), device='cpu')