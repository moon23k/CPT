import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.model_name = config.model_name
        self.data = self.read_dataset(split)

    @staticmethod
    def read_dataset(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']
        
        if self.model_name == 'transformer':
            trg_input, trg_output = trg[:-1], trg[1:]
        else:
            trg_input, trg_output = trg, trg[1:]
        
        return src, trg_input, trg_output



def load_dataloader(config, split):
    global pad_idx
    pad_idx = config.pad_idx    


    def _collate_fn(batch):
        src_batch, trg_input_batch, trg_output_batch = [], [], []
        
        for src, trg_input, trg_output in batch:
            src_batch.append(torch.LongTensor(src))
            trg_input_batch.append(torch.LongTensor(trg_input))
            trg_output_batch.append(torch.LongTensor(trg_output))
        
        src_batch = pad_sequence(src_batch, 
                                 batch_first=True, 
                                 padding_value=pad_idx)
        
        trg_input_batch = pad_sequence(trg_input_batch, 
                                       batch_first=True, 
                                       padding_value=pad_idx)
        
        trg_output_batch = pad_sequence(trg_output_batch, 
                                        batch_first=True, 
                                        padding_value=pad_idx)
        
        return {'src': src_batch, 
                'trg_input': trg_input_batch, 
                'trg_output': trg_output_batch}


    dataset = Dataset(config, split)
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False, 
                      collate_fn=_collate_fn, 
                      num_workers=2)