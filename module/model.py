import os, torch
import torch.nn as nn
from model import Encoder, Transformer



def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)




def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")




def load_model(config):
    mode = config.mode
    pt_obj = config.pt_obj
    ckpt = config.pt_ckpt if mode == 'train' else config.ckpt

    if mode == 'pretrain' and pt_obj == 'masked':
        model = Encoder(config)
        init_weights(model)
        print(f"Initialized Encoder Model for {pt_obj.upper()} Language Modeling PreTraining has Loaded")


    model = Transformer(config)
    init_weights(model)
    print("Initialized Model has Loaded")

    if mode == 'train' and pt_obj == 'masked':
        assert os.path.exists(ckpt)
        pt_state_dict = torch.load(ckpt)
        for name, param in pt_state_dict['model_state_dict'].items():
            model.state_dict()[f"encoder.{name}"] = param
        print(f"PreTrained Encoder States have loaded from {ckpt}")
    
    else:
        assert os.path.exists(ckpt)
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"{'PreTrained' if mode == 'train' else 'Trained'} Model States have loaded from {ckpt}")

    print_model_desc(model)
    return model.to(config.device)