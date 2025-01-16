import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import PIL

DEVICE = 'mps'

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 条件类别嵌入
        self.class_embed = nn.Embedding(num_classes, 128)
        
        self.cond_proj = nn.Sequential(
            nn.Linear(128, 256),  # 将128维扩展到256维匹配特征图
            nn.ReLU()
        )
        
        # 下采样路径
        self.down1 = nn.Conv2d(1, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # 上采样路径
        self.up1 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 1, 3, padding=1)
        
    def forward(self, x, t, labels):
        # 嵌入时间和类别
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        c_emb = self.class_embed(labels)
        
        # 合并条件信息
        cond = self.cond_proj(t_emb + c_emb).unsqueeze(-1).unsqueeze(-1)
        
        # 下采样
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        d3 = F.relu(self.down3(d2))
        
        # print(d3.shape, cond.shape)
        # 注入条件
        d3 = d3 + cond
        
        # 上采样
        u1 = F.relu(self.up1(d3))
        u2 = F.relu(self.up2(u1 + d2))
        out = self.up3(u2 + d1)
        
        return out

class DiffusionModel:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
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
        model.eval()
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

def train_diffusion():
    # 配置
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model = ConditionalUNet().to(device)
    diffusion = DiffusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 训练循环
    for epoch in range(100):
        for batch, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 随机时间步
            t = torch.randint(0, diffusion.num_steps, (images.shape[0],)).to(device)
            
            # 前向扩散
            noisy_images, noise = diffusion.forward_diffusion(images, t)
            
            # 预测噪声
            predicted_noise = model(noisy_images, t, labels)
            
            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch}, Loss {loss.item():.4f}')
                sample_model(model, labels)

def sample_model(model, labelIn):
    # 配置
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    diffusion = DiffusionModel()
    
    # 加载模型
    # model.load_state_dict(torch.load('model.pth'))
    
    # 采样
    with torch.no_grad():
        samples = diffusion.sample(model, labelIn.shape[0], labelIn, device)
        
        PIL.Image.fromarray(
            ((samples[0].cpu().numpy().squeeze() + 1) * 127.5).astype('uint8')
        ).show()
    
    return samples

if __name__ == "__main__":
    train_diffusion()