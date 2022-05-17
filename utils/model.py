import torch
import torch.nn as nn
from models.seq2seq.module import Seq2Seq
from models.attention.module import Seq2Seq_Attn
from models.transformer.module import Transformer




def init_uniform(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)



def init_normal(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)



def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(config):
    if config.model == 'seq2seq':
        model = Seq2Seq(config)
        model.apply(init_uniform)
    
    elif config.model == 'attention':
        model = Seq2Seq_Attn(config)
        model.apply(init_normal)
    
    elif config.model == 'transformer':
        model = Transformer(config)
        model.apply(init_xavier)    
    

    model.to(config.device)
    print(f'{config.model} model has loaded.\nThe model has {count_params(model):,} trainable parameters')
    print()
    
    return model