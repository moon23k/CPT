import torch, math, time
import torch.nn as nn
from modules.search import RNNSearch, TransSearch



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        self.model = model
        self.model_name = config.model_name
        
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.device = config.device
        self.batch_size = config.batch_size        
        self.output_dim = config.output_dim
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(self.device)

        if self.model_name != 'transformer':
            self.search = RNNSearch(config, self.model, tokenizer)
        elif self.model_name == 'transformer':
            self.search = TransSearch(config, self.model, tokenizer)            


    def test(self):
        self.model.eval()
        tot_len = len(self.dataloader)
        tot_loss = 0.0
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg_input = batch['trg_input'].to(self.device)
                trg_output = batch['trg_output'].to(self.device)

                if self.model_name== 'transformer':
                    logit = self.model(src, trg_input)
                else:
                    logit = self.model(src, trg_input, teacher_forcing_ratio=0.0)

                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg_output.contiguous().view(-1)).item()

                tot_loss += loss
            tot_loss /= tot_len
        
        print(f'Test Results on {self.model_name} model')
        print(f">> Test Loss: {tot_loss:.3f} | Test PPL: {math.exp(tot_loss):.2f}\n")


        
    def inference_test(self):
        self.model.eval()
        batch = next(iter(self.dataloader))
        input_batch = batch['src'].to(self.device)
        label_batch = batch['trg_input'].to(self.device)

        inference_dicts = []
        for i in range(3):
            temp_dict = dict()

            input_seq, label_seq = input_batch[i], label_batch[i] 
            input_seq = self.tokenizer.decode(input_seq.tolist()) 
            label_seq = self.tokenizer.decode(label_seq.tolist())                        

            temp_dict['input_seq'] = input_seq
            temp_dict['label_seq'] = label_seq

            temp_dict['greedy_out'] = self.search.greedy_search(input_seq)
            temp_dict['beam_out'] = self.search.beam_search(input_seq)
            
            inference_dicts.append(temp_dict)


        print(f'Inference Test on {self.model_name} model')
        for d in inference_dicts:
            print(f">> Input  Sequence: {d['input_seq']}")
            print(f">> Label  Sequence: {d['label_seq']}")
            print(f">> Greedy Sequence: {d['greedy_out']}")
            print(f">> Beam   Sequence: {d['beam_out']}\n")
