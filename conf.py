# support object config, dict config, and yaml config
import os
import yaml
import json
import argparse

class Config(object):
    # dataset config
    dataset = 'bench'
    # data_dir = '/Users/zhangxu/Desktop/ssvep/bench/'
    num_class = 40
    data_t0 = 50
    init_t0 = 125 + 35
    multiplex = 5
    delt_t = 10
    batch_size = 40
    channel = []
    for i in [48, 54, 55, 56, 57, 58, 61, 62, 63]:
        channel.append(i-1)
    idx = 0
    val_person_id = None
    worker = 8
    
    
    # model config
    task_name = 'short_term_forecast' # 'classification' # 'short_term_forecast'
    seq_len = data_t0
    pred_len = data_t0
    stride = 5
    patch_len = 10
    factor = 2
    n_heads = 6
    activation = 'relu'
    dropout = 0.25
    e_layers = 5
    enc_in = len(channel)
    d_model = 64
    d_ff = 1024
    
    # optimizer config
    lr = 0.001
    weight = 1e-5
    momentum = 0.9
    beta = (0.9, 0.999)
    epoch = 100
    
    # log config
    device = 'cuda:0'
    log_dir = '/home/zhangxu/logger'
    model_dir = '/home/zhangxu/saved_model'
    model_name = 'model.pth'


def load_yaml(config_path):
    # transfer yaml into class
    config = Config()
    with open(config_path, 'r') as f:
        yaml_config = yaml.load(f)
    for k, v in yaml_config.items():
        setattr(config, k, v)
    return config

def load_json(config_path):
    # transfer json into class
    config = Config()
    with open(config_path, 'r') as f:
        json_config = json.load(f)
    for k, v in json_config.items():
        setattr(config, k, v)
    return config

def from_dict(config_dict):
    # transfer dict into class
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)
    return config

def to_dict(config):
    # transfer class into dict
    config_dict = {}
    for k, v in config.__dict__.items():
        config_dict[k] = v
    return config_dict

def to_json(config, config_path):
    # transfer class into json
    config_dict = to_dict(config)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
def to_yaml(config, config_path):
    # transfer class into yaml
    config_dict = to_dict(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
        
def make_args():
    # make args for parser
    parser = argparse.ArgumentParser()
    # add arguments by Config
    for k, v in Config.__dict__.items():
        if not k.startswith('__'):
            parser.add_argument('--' + k, type=type(v), default=v)
    args = parser.parse_args()
    
    # transfer args into class
    config = Config()
    for k, v in args.__dict__.items():
        setattr(config, k, v)
    return config