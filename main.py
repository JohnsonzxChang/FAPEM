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
from Patch_SSVEP import SSVEPModel

def accuracy(y_pred, y):
    y_pred = th.argmax(y_pred, dim=1)
    return th.sum(y_pred == y).item() / len(y)

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
        self.logger = SummaryWriter(comment=comment)
        
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
    

def main():
    # load config
    # conf = load_yaml('conf.yaml')
    conf = Config()
    # init pipline
    pipline = ClassificationPipline(conf, comment='classification')
    # load data
    pipline._load_data(data_loader_fcn=get_dataloader)
    # model, optimizer, loss
    pipline._model_optimizer_loss(SSVEPModel, nn.CrossEntropyLoss(), optim.NAdam)
    # load ckpt
    # pipline._load_ckpt()
    # early stop
    pipline._conf_early_stop(patience=10, save_ckpt=True)
    # train and validate
    for e in tqdm(range(conf.epoch)):
        pipline.train(e)
        stop, local_loss, local_acc = pipline.validate(e)
        if stop:
            print('Early stop at epoch %d, val loss %.4f, val acc %.4f' % (e, local_loss, local_acc))
            break
    pipline.logger.close()
    
if __name__ == '__main__':
    main()