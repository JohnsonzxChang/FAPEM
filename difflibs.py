import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm
from denoising_diffusion_pytorch import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import cycle, num_to_groups, has_int_squareroot, exists
from dt_simple import *

class Trainer_Diffusions(object):
    def __init__(
        self, conf,
        diffusion_model: GaussianDiffusion1D
    ):
        super().__init__()
        assert isinstance(conf, Config)
        self.conf = conf
        self.logger = SummaryWriter(comment='diffusion', log_dir='./results')
        train_batch_size = conf.batch_size
        gradient_accumulate_every = 1
        train_lr = conf.lr
        train_num_steps = conf.epoch
        ema_update_every = 10
        ema_decay = 0.995
        adam_betas = conf.beta
        save_and_sample_every = conf.diff_num_step
        num_samples = 25
        results_folder = './results'
        amp = False
        mixed_precision_type = 'fp16'
        split_batches = True
        max_grad_norm = 1.0

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        dl = get_dataloader(conf)['trn_dataloader']
        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        e = 0
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch_data = next(self.dl) #.to(device)
                    x = batch_data['data'].squeeze(0).to(self.conf.device)
                    ids = batch_data['id'].squeeze(0).to(self.conf.device)
                    y = batch_data['label'].squeeze(0).to(self.conf.device)
                    # x_aux = batch_data['data_aux'].squeeze(0).to(self.conf.device).reshape((-1, 3*9, 50))
                    
                    loss = 0.
                    with self.accelerator.autocast():
                        for chn in range(x.shape[2]):
                            loss += self.model(x[:,:,chn,:])
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                self.logger.add_scalar('loss', total_loss, e)
                e += 1
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        self.logger.close()


if __name__ == '__main__':
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = 3
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 48,
        timesteps = 1000,
        objective = 'pred_v'
    )

    from torchinfo import summary
    summary(diffusion, (2,3,48))

    trainer = Trainer_Diffusions(
        Config(),
        diffusion
    )
    trainer.train()

    # after a lot of training

    sampled_seq = diffusion.sample(batch_size = 4)
    sampled_seq.shape # (4, 32, 128)