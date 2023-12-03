import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    Trainer,
    Tester,
    Generator
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.search_method = args.search
        
        self.ckpt = f"ckpt/{self.task}/model.pt"
        self.pt_ckpt = f"ckpt/{self.task}/pt_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)


        #여기가 모델을 로드하는 과정에서의 조건값 세팅
        self.pretrain_encoder = True if self.pt_strategy != 'none' else False
        self.pretrain_decoder = True if self.pt_strategy == 'seq2seq' else False


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'pretrain':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()        

    elif config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        generator = Generator(config, model, tokenizer)
        generator.inference()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-pt_strategy', default='none', required=False)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task.lower() in ['translation', 'dialogue', 'summarization']
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']
    assert args.pt_strategy.lower() in ['none', 'encoder', 'seq2seq']
    assert args.search.lower() in ['greedy', 'beam']

    os.makedirs(f"ckpt/{args.pt_strategy}", exist_ok=True)
    main(args)