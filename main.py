import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from conf import *
from dt_simple import *
from mills import *
from Patch_SSVEP import SSVEPModel
from GT_SSVEP import Model
from ts_SSVEP import SSVEPModelAll
from originCNN import SSVEP_Net
from vani_SSVEP import VanillaTransformer
from diff import ConditionalUNet, DiffusionModel

class Pipline(object):
    def __init__(self, conf, comment):
        if type(conf) == dict:
            self.conf = from_dict(conf)
        elif type(conf) == str:
            if conf.endswith('yaml'):
                self.conf = load_yaml(conf)
            elif conf.endswith('json'):
                self.conf = load_json(conf)
            else:
                raise ValueError('Unsupported config file format')
        self.conf = conf
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.logger = SummaryWriter(comment=comment, log_dir=self.conf.log_dir)
        
    def _load_data(self):
        raise NotImplementedError
    
    def _model_optimizer_loss(self):
        raise NotImplementedError
    
    def _load_ckpt(self):
        raise NotImplementedError
    
    def _save_ckpt(self):
        raise NotImplementedError
    
class ClassificationPipline(Pipline):
    def __init__(self, conf, comment):
        super(ClassificationPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        self.criterion = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr, }) # 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras) # , betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _cla_flow(self, batch):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return loss, y_pred, y
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, y_pred, y = self._cla_flow(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss', total_loss/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, y_pred, y = self._cla_flow(batch)
                total_loss += loss.item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss = total_loss/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss', local_loss, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss < self.best_val_loss or self.best_val_loss == np.inf:
                        self.best_val_loss = local_loss
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss, local_acc
        return False, local_loss, local_acc
    
class ForcastPipline(Pipline):
    def __init__(self, conf, comment):
        super(ForcastPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        
        # A = torch.tensor(np.random.random((9, 9))).to(self.conf.device)
        # self.model = Model(num_node=9, input_dim=3, hidden_dim=8, output_dim=32, embed_dim=64, 
        #             cheb_k=3, horizon=50, num_layers=4, heads=4, A=A, kernel_size=3, max_len=50).to(self.conf.device)
        
        self.criterion = nn.L1Loss()
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr, }) # 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras) # , betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, f'{self.conf.model_name}')
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _cla_flow(self, batch):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        x_aux = batch['data_aux'].squeeze(0).to(self.conf.device).mean(dim=1)
        x_pred = self.model(x)
        assert x_pred.shape == x_aux.shape, 'Shape mismatch, x_pred %s, x_aux %s' % (x_pred.shape, x_aux.shape)
        loss = self.criterion(x_pred, x_aux)
        return loss, x_pred, x_aux
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss = 0
        total_B = 0
        # all_y_pred = None
        # all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, y_pred, y = self._cla_flow(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_B += y.shape[0]
            # if e % deltE == 0:
            #     if all_y_pred is None:
            #         all_y_pred = y_pred
            #         all_y = y
            #     else:
            #         all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
            #         all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss', total_loss/total_B, e)
            # self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss = 0
        total_B = 0
        # all_y_pred = None
        # all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, y_pred, y = self._cla_flow(batch)
                total_loss += loss.item()
                total_B += y.shape[0]
                # if e % deltE == 0:
                #     if all_y_pred is None:
                #         all_y_pred = y_pred
                #         all_y = y
                #     else:
                #         all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                #         all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss = total_loss/total_B
                # local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss', local_loss, e)
                # self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss < self.best_val_loss or self.best_val_loss == np.inf:
                        self.best_val_loss = local_loss
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss, None
        return False, local_loss, None 
    
    def view_result(self, n):
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, x_pred, x = self._cla_flow(batch)
            # plot last batch
            plt.figure()
            plt.plot(x.cpu().numpy()[n,:,:].T, label='x', color='r')
            plt.plot(x_pred.cpu().numpy()[n,:,:].T, label='x_pred', color='b')
            plt.legend()
            plt.show()
        print('show done')


class MixPipline(Pipline):
    def __init__(self, conf, comment):
        super(MixPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        
        # A = torch.tensor(np.random.random((9, 9))).to(self.conf.device)
        # self.model = Model(num_node=9, input_dim=3, hidden_dim=8, output_dim=32, embed_dim=64, 
        #             cheb_k=3, horizon=50, num_layers=4, heads=4, A=A, kernel_size=3, max_len=50).to(self.conf.device)
        
        self.criterion1 = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.criterionCla = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr, }) # 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras) # , betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, f'{self.conf.model_name}')
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _mix_flow(self, batch, e):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        x_aux = batch['data_aux'].squeeze(0).to(self.conf.device).mean(dim=1)
        x_pred = self.model.f1(x)
        y_pred = self.model.f2(x)
        assert x_pred.shape == x_aux.shape, 'Shape mismatch, x_pred %s, x_aux %s' % (x_pred.shape, x_aux.shape)
        assert x_pred.shape[1] == 9, 'Shape mismatch, x_pred %s' % (x_pred.shape)
        loss1, loss2 = 0, 0
        for i in range(x_pred.shape[1]):
            loss1 += self.criterion1(x_pred[:,i,:], x_aux[:,i,:]) * 0.9
            loss2 += self.criterion2(x_pred[:,i,:], x_aux[:,i,:]) * 0.1
        loss1 = loss1 / x_pred.shape[1]
        loss2 = loss2 / x_pred.shape[1]
        loss3 = self.criterionCla(y_pred, y) * (e > self.conf.warmup) * 1.0
        return [loss1, loss2, loss3], [x_pred, x_aux], [y_pred, y]
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, _, [y_pred, y] = self._mix_flow(batch, e)
            (loss[0] + loss[1] + loss[2]).backward() #  + loss[2]
            self.optimizer.step()
            total_loss1 += loss[0].item()
            total_loss2 += loss[1].item()
            total_loss3 += loss[2].item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss1', total_loss1/total_B, e)
            self.logger.add_scalar('train/loss2', total_loss2/total_B, e)
            self.logger.add_scalar('train/loss3', total_loss3/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, _, [y_pred, y] = self._mix_flow(batch, e)
                self.optimizer.step()
                total_loss1 += loss[0].item()
                total_loss2 += loss[1].item()
                total_loss3 += loss[2].item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss1 = total_loss1/total_B
                local_loss2 = total_loss2/total_B
                local_loss3 = total_loss3/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss1', local_loss1, e)
                self.logger.add_scalar('val/loss2', local_loss2, e)
                self.logger.add_scalar('val/loss3', local_loss3, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss3 <= self.best_val_loss or self.best_val_loss == np.inf or e <= self.conf.warmup*2:
                        self.best_val_loss = local_loss3
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss3, None
        return False, local_loss3, None 
    
    def view_result(self, n):
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, [x_pred, x_aux], [y_pred, y] = self._mix_flow(batch, 0)
                self.optimizer.step()
            # plot last batch
            plt.figure()
            for j in range(9):
                plt.subplot(9,1,j+1)
                plt.plot(x_aux.cpu().numpy()[n,j,:].T, label='x', color='r')
                plt.plot(x_pred.cpu().numpy()[n,j,:].T, label='x_pred', color='b')
            plt.legend()
            plt.show()
        print('show done')

class BenchmarkPipline(Pipline):
    def __init__(self, conf, comment):
        super(BenchmarkPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        self.criterion = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr, 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras, betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _cla_flow(self, batch):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        y_pred = self.model(x, ids)
        loss = self.criterion(y_pred, y)
        return loss, y_pred, y
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, y_pred, y = self._cla_flow(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss', total_loss/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, y_pred, y = self._cla_flow(batch)
                total_loss += loss.item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss = total_loss/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss', local_loss, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss < self.best_val_loss or self.best_val_loss == np.inf:
                        self.best_val_loss = local_loss
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss, local_acc
        return False, local_loss, local_acc
    
class StepPipline(Pipline):
    def __init__(self, conf, comment):
        super(StepPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        
        # A = torch.tensor(np.random.random((9, 9))).to(self.conf.device)
        # self.model = Model(num_node=9, input_dim=3, hidden_dim=8, output_dim=32, embed_dim=64, 
        #             cheb_k=3, horizon=50, num_layers=4, heads=4, A=A, kernel_size=3, max_len=50).to(self.conf.device)
        
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterionCla = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr})#, 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras)#, betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, f'{self.conf.model_name}')
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _mix_flow(self, batch, e):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        x_aux = batch['data_aux'].squeeze(0).to(self.conf.device).reshape((-1, 3*9, 50))
        # y_pred, x_pred = self.model(x)
        x_pred = self.model(x)
        y_pred = y
        x_pred = x_pred.reshape((-1, 9*3, 50))
        assert x_pred.shape == x_aux.shape, 'Shape mismatch, x_pred %s, x_aux %s' % (x_pred.shape, x_aux.shape)
        # assert x_pred.shape[1] == 9, 'Shape mismatch, x_pred %s' % (x_pred.shape)
        loss1 = 0
        loss2 = 0
        for i in range(x_pred.shape[1]):
            loss1 += self.criterion2(x_pred[:,i,:], x_aux[:,i,:]) * 1.0#10
            loss2 += self.criterion2(x_pred[:,i,1:]-x_pred[:,i,:-1], x_aux[:,i,1:]-x_aux[:,i,:-1]) * 1e-1#05
        loss1 = loss1 / x_pred.shape[1]
        # loss2 = 0.1 * th.mean(th.relu(th.log(th.std(x_aux,dim=-1)) - th.log(th.std(x_pred,dim=-1))))
        loss3 = th.tensor(0) # self.criterionCla(y_pred, y) * 0.0 #* (e > self.conf.warmup) * 1.0
        return [loss1, loss2, loss3], [x_pred, x_aux], [y_pred, y]
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, _, [y_pred, y] = self._mix_flow(batch, e)
            (loss[0]+ loss[1]).backward() #   + loss[2]
            self.optimizer.step()
            total_loss1 += loss[0].item()
            total_loss2 += loss[1].item()
            total_loss3 += loss[2].item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss1', total_loss1/total_B, e)
            self.logger.add_scalar('train/loss2', total_loss2/total_B, e)
            self.logger.add_scalar('train/loss3', total_loss3/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, _, [y_pred, y] = self._mix_flow(batch, e)
                self.optimizer.step()
                total_loss1 += loss[0].item()
                total_loss2 += loss[1].item()
                total_loss3 += loss[2].item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss1 = total_loss1/total_B
                local_loss2 = total_loss2/total_B
                local_loss3 = total_loss3/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss1', local_loss1, e)
                self.logger.add_scalar('val/loss2', local_loss2, e)
                self.logger.add_scalar('val/loss3', local_loss3, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss1 < self.best_val_loss or self.best_val_loss == np.inf or e <= self.conf.warmup:
                        self.best_val_loss = local_loss1
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss1, None
        return False, local_loss1, None 
    
    def view_result(self, n):
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, [x_pred, x_aux], [y_pred, y] = self._mix_flow(batch, 0)
                # self.optimizer.step()
            # plot last batch
            plt.figure()
            for k in range(3):
                for j in range(9):
                    plt.subplot(9,3,j*3+k+1)
                    plt.plot(x_aux.cpu().numpy()[n,j*3+k,:].T, label='x', color='r')
                    plt.plot(x_pred.cpu().numpy()[n,j*3+k,:].T, label='x_pred', color='b')
                    plt.ylabel(f'{j},{k}')
                    plt.legend()
            plt.show()
        print('show done')

class ImputationPipline(Pipline):
    def __init__(self, conf, comment):
        super(ImputationPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterionCla = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr})#, 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras)#, betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, f'{self.conf.model_name}')
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _mix_flow(self, batch, e):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        # x_aux = batch['data_aux'].squeeze(0).to(self.conf.device).reshape((-1, 3*9, 50))
        # 生成mask
        B,C,M,T = x.shape
        if self.conf.mask_mode == 'random':
            # 随机mask
            mask = torch.bernoulli(torch.ones(B,C,1,T) * self.conf.mask_ratio).to(x.device)
            mask = mask.repeat(1, 1, M, 1)
        elif self.conf.mask_mode == 'continous':
            # 连续mask
            mask = torch.zeros_like(x)
            # 配置连续mask的长度范围
            min_mask_len = max(1, int(T * (1.0- self.conf.mask_ratio)))
            max_mask_len = int(T * 1.0)
            
            for b in range(B):
                for c in range(C):
                    for m in range(M):
                        # 随机选择mask长度
                        mask_len = torch.randint(min_mask_len, max_mask_len+1, (1,)).item()
                        
                        # 随机选择起始位置
                        start_idx = torch.randint(0, T - mask_len + 1, (1,)).item()
                        
                        # 生成连续mask
                        mask[b, c, m, start_idx:start_idx+mask_len] = 1
        else:
            raise ValueError('Unknown mask method')
        
        # 生成masked输入
        x_masked = x * (1 - mask) # + noise * mask 是否要加噪声？
        x_pred = self.model(x_masked)
        y_pred = y
        loss1 = self.criterion2(x_pred * mask, x * mask) * 1.0#10
        loss2 = self.criterion2((x_pred[:,:,:,1:]-x_pred[:,:,:,:-1]) * mask[:,:,:,:-1], (x[:,:,:,1:]-x[:,:,:,:-1]) * mask[:,:,:,:-1]) * 1e-1#05
        # loss2 = 0.1 * th.mean(th.relu(th.log(th.std(x_aux,dim=-1)) - th.log(th.std(x_pred,dim=-1))))
        loss3 = th.tensor(0) # self.criterionCla(y_pred, y) * 0.0 #* (e > self.conf.warmup) * 1.0
        return [loss1, loss2, loss3], [x_pred, x], [y_pred, y]
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, _, [y_pred, y] = self._mix_flow(batch, e)
            (loss[0]+ loss[1]).backward() #   + loss[2]
            self.optimizer.step()
            total_loss1 += loss[0].item()
            total_loss2 += loss[1].item()
            total_loss3 += loss[2].item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss1', total_loss1/total_B, e)
            self.logger.add_scalar('train/loss2', total_loss2/total_B, e)
            self.logger.add_scalar('train/loss3', total_loss3/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, _, [y_pred, y] = self._mix_flow(batch, e)
                self.optimizer.step()
                total_loss1 += loss[0].item()
                total_loss2 += loss[1].item()
                total_loss3 += loss[2].item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss1 = total_loss1/total_B
                local_loss2 = total_loss2/total_B
                local_loss3 = total_loss3/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss1', local_loss1, e)
                self.logger.add_scalar('val/loss2', local_loss2, e)
                self.logger.add_scalar('val/loss3', local_loss3, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss1 < self.best_val_loss or self.best_val_loss == np.inf or e <= self.conf.warmup:
                        self.best_val_loss = local_loss1
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss1, None
        return False, local_loss1, None 
    
    def view_result(self, n):
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, [x_pred, x_aux], [y_pred, y] = self._mix_flow(batch, 0)
                # self.optimizer.step()
            # plot last batch
            plt.figure()
            for k in range(3):
                for j in range(9):
                    plt.subplot(9,3,j*3+k+1)
                    plt.plot(x_aux.cpu().numpy()[n,k,j,:].T, label='x', color='r')
                    plt.plot(x_pred.cpu().numpy()[n,k,j,:].T, label='x_pred', color='b')
                    plt.ylabel(f'{j},{k}')
                    plt.legend()
            plt.show()
        print('show done')

class UnetDiffPipline(Pipline):
    def __init__(self, conf, comment):
        super(UnetDiffPipline, self).__init__(conf, comment)
        self.patience = None
    
    def _load_data(self, data_loader_fcn):
        self.loader = data_loader_fcn(self.conf)
        assert 'trn_dataloader' in self.loader.keys(), 'trn_dataloader not found'
        assert 'val_dataloader' in self.loader.keys(), 'val_dataloader not found'
    
    def _model_optimizer_loss(self, model_fcn, criterion_instance, optimizer_fcn):
        self.model = model_fcn(self.conf).to(self.conf.device)
        self.diffusion = DiffusionModel(self.conf) # .to(self.conf.device)
        # A = torch.tensor(np.random.random((9, 9))).to(self.conf.device)
        # self.model = Model(num_node=9, input_dim=3, hidden_dim=8, output_dim=32, embed_dim=64, 
        #             cheb_k=3, horizon=50, num_layers=4, heads=4, A=A, kernel_size=3, max_len=50).to(self.conf.device)
        
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterionCla = criterion_instance
        paras = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            paras.append({'params': param, 'lr': self.conf.lr})#, 'weight_decay': self.conf.weight})
        self.optimizer = optimizer_fcn(paras)#, betas=self.conf.beta, momentum_decay=self.conf.momentum)
    
    def _load_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, f'{self.conf.model_name}')
        if os.path.exists(ckpt_path):
            ckpt = th.load(ckpt_path, map_location=self.conf.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise ValueError('No checkpoint found')
    
    def _save_ckpt(self):
        ckpt_path = os.path.join(self.conf.model_dir, self.conf.model_name)
        th.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 
                ckpt_path)
        
    def _mix_flow(self, batch, e):
        x = batch['data'].squeeze(0).to(self.conf.device)
        ids = batch['id'].squeeze(0).to(self.conf.device)
        y = batch['label'].squeeze(0).to(self.conf.device)
        x_aux = batch['data_aux'].squeeze(0).to(self.conf.device).reshape((-1, 3*9, 50))
        
        # 随机时间步
        t = torch.randint(0, self.diffusion.num_steps, (x.shape[0],)).to(self.conf.device)
        
        # 前向扩散
        noisy_images, noise = self.diffusion.forward_diffusion(x, t)
        
        # 预测噪声
        predicted_noise = self.model(noisy_images, t, y)
        
        # 计算损失
        loss1 = FF.mse_loss(predicted_noise, noise)
        
        all_ref = self.diffusion.sample_all_class(self.model, range(self.conf.num_class), x.shape, self.conf.device)
        x_pred = self.model.combine_all(x)# or x
        all_ref = self.model.combine(all_ref)
        y_pred = self.model.corr(x_pred, all_ref)
        # print(y_pred.shape, y.shape)
        assert y_pred.shape[-1] == 40, 'Shape mismatch, y_pred %s' % (y_pred.shape)
        loss2 = self.criterionCla(y_pred, y)
        return [loss1, loss2], [x_pred, all_ref], [y_pred, y]
    
    def _conf_early_stop(self, patience=10, save_ckpt=True):
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.save_ckpt = save_ckpt
    
    def train(self, e, deltE=1):
        self.model.train()
        total_loss1 = 0
        total_loss2 = 0
        # total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        for i, batch in enumerate(self.loader['trn_dataloader']):
            self.optimizer.zero_grad()
            loss, _, [y_pred, y] = self._mix_flow(batch, e)
            (loss[0]+ loss[1]).backward() #   + loss[2]
            self.optimizer.step()
            total_loss1 += loss[0].item()
            total_loss2 += loss[1].item()
            # total_loss3 += loss[2].item()
            total_B += y.shape[0]
            if e % deltE == 0:
                if all_y_pred is None:
                    all_y_pred = y_pred
                    all_y = y
                else:
                    all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                    all_y = th.cat((all_y, y), dim=0)
        if e % deltE == 0:
            self.logger.add_scalar('train/loss1', total_loss1/total_B, e)
            self.logger.add_scalar('train/loss2', total_loss2/total_B, e)
            # self.logger.add_scalar('train/loss3', total_loss3/total_B, e)
            self.logger.add_scalar('train/acc', accuracy(all_y_pred, all_y), e)
    
    def validate(self, e, deltE=1):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        # total_loss3 = 0
        total_B = 0
        all_y_pred = None
        all_y = None
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, _, [y_pred, y] = self._mix_flow(batch, e)
                self.optimizer.step()
                total_loss1 += loss[0].item()
                total_loss2 += loss[1].item()
                # total_loss3 += loss[2].item()
                total_B += y.shape[0]
                if e % deltE == 0:
                    if all_y_pred is None:
                        all_y_pred = y_pred
                        all_y = y
                    else:
                        all_y_pred = th.cat((all_y_pred, y_pred), dim=0)
                        all_y = th.cat((all_y, y), dim=0)
            if e % deltE == 0:
                local_loss1 = total_loss1/total_B
                local_loss2 = total_loss2/total_B
                # local_loss3 = total_loss3/total_B
                local_acc = accuracy(all_y_pred, all_y)
                self.logger.add_scalar('val/loss1', local_loss1, e)
                self.logger.add_scalar('val/loss2', local_loss2, e)
                # self.logger.add_scalar('val/loss3', local_loss3, e)
                self.logger.add_scalar('val/acc', local_acc, e)
                if self.patience is not None:
                    if local_loss1 < self.best_val_loss or self.best_val_loss == np.inf or e <= self.conf.warmup:
                        self.best_val_loss = local_loss1
                        self.counter = 0
                        if self.save_ckpt:
                            self._save_ckpt()
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            return True, local_loss1, None
        return False, local_loss1, None 
    
    def view_result(self, n):
        with th.no_grad():
            for i, batch in enumerate(self.loader['val_dataloader']):
                loss, [x_pred, all_ref], [y_pred, y] = self._mix_flow(batch, 0)
                # self.optimizer.step()
            # plot last batch
            plt.figure()
            for k in range(3):
                for j in range(3):
                    plt.subplot(3,3,j*3+k+1)
                    plt.plot(x_pred.cpu().numpy()[n,k*3+j,:].T, label='x_pred', color='r')
                    plt.plot(all_ref.cpu().numpy()[n,k*3+j,:].T, label='all_ref', color='g')
                    
                    plt.ylabel(f'{j},{k}')
                    plt.legend()
            plt.show()
        print('show done')


def main(train=True):
    # set seed
    # setup_seed(10086)
    # load config
    # conf = load_yaml('conf.yaml')
    conf = Config()
    # init pipline
    if conf.task_name == 'short_term_forecast':
        pipline = ForcastPipline(conf, comment='short_term_forecast')
        netClass = SSVEPModel
    elif conf.task_name == 'classification':
        pipline = ClassificationPipline(conf, comment='classification')
        netClass = SSVEPModel
    elif conf.task_name == 'Mixture':
        pipline = MixPipline(conf, comment='classification')
        netClass = SSVEPModel
    elif conf.task_name == 'Step':
        pipline = StepPipline(conf, comment='classification')
        netClass = VanillaTransformer # SSVEPModelAll
    elif conf.task_name == 'Benchmark':
        pipline = BenchmarkPipline(conf, comment='classification')
        netClass = SSVEP_Net
    elif conf.task_name == 'Imputation':
        pipline = ImputationPipline(conf, comment='Imputation')
        netClass = VanillaTransformer # SSVEPModelAll
    elif conf.task_name == 'Diffusion':
        pipline = UnetDiffPipline(conf, comment='Diffusion')
        netClass = ConditionalUNet # SSVEPModelAll
    # load data
    pipline._load_data(data_loader_fcn=get_dataloader)
    # model, optimizer, loss
    pipline._model_optimizer_loss(netClass, CrossEntropyLabelSmooth(40, 0.1), optim.Adam) # CrossEntropyLabelSmooth(40, 0.1)
    # load ckpt
    # pipline._load_ckpt()
    # early stop
    pipline._conf_early_stop(patience=20, save_ckpt=True)
    # train and validate
    if train:
        for e in tqdm(range(conf.epoch)):
            pipline.train(e)
            stop, local_loss, local_acc = pipline.validate(e)
            if stop:
                print(f'Early stop at epoch {e}, val loss {local_loss}, val acc {local_acc}')
                break
        pipline.logger.close()
    else:
        pipline._load_ckpt()
    return pipline
    
if __name__ == '__main__':
    pipline = main(True)
    # input data number
    if (hasattr(pipline, 'view_result')):
        while True:
            n = input('Input data number to view result: ')
            try:
                n = int(n)
            except:
                print('Invalid input, stop')
                break
            pipline.view_result(int(n))