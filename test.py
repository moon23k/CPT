import time
import argparse

import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score
import sentencepiece as spm

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import seq_eval, trans_eval 
from utils.util import Config, epoch_time, set_seed





def seq_bleu(model, dataloader, tokenizer):
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].to(config.device), batch[1].to(config.device)
            pred = model(src, trg)

            pred = pred.argmax(-1).tolist()
            trg = [[str(ids) for ids in seq] for seq in trg.tolist()]
            bleu = bleu_score(pred, trg)

    return bleu




def trans_bleu(model, dataloader, tokenizer):
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].to(config.device), batch[1].to(config.device)
            pred = model(src, trg)

            pred = pred.argmax(-1).tolist()
            trg = [[str(ids) for ids in seq] for seq in trg.tolist()]
            bleu = bleu_score(pred, trg)

    return bleu




def run(config):
    chk_file = f"checkpoints/{config.model}_states.pt"
	test_dataloader = get_dataloader('test', config)

    #Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

    #Load Model
    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)

    start_time = time.time()
    if config.model == 'transformer':
        test_loss = trans_eval(model, test_dataloader, criterion, config)
        test_bleu = trans_bleu(model, test_dataloader, tokenizer)
    else:
        test_loss = seq_eval(model, test_dataloader, criterion, config)
        test_bleu = seq_bleu(model, test_dataloader, tokenizer)
    end_time = time.time()
    test_mins, test_secs = epoch_time(start_time, end_time)
    
    print(f"Test Loss: {test_loss} / Test BLEU Score: {test_bleu} / Time: {test_mins}min {test_secs}sec")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()

    assert args.model in ['seq2seq', 'attention', 'transformer']

    set_seed()
    config = Config(args)
    run(config)