import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.model_name = config.model_name
        self.data = self.read_dataset(split)


    def read_dataset(self, split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        if self.model_name == 'transformer':
            trg = self.data[idx]['trg'][:-1]
        else:
            trg = self.data[idx]['trg']
        return src, trg 



def rnn_collate(batch):
    src_batch, trg_batch = [], []
    
    for src, trg in batch:
        src_batch.append(torch.LongTensor(src))
        trg_batch.append(torch.LongTensor(trg))
        
    src_batch = pad_sequence(src_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    trg_batch = pad_sequence(trg_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    return {'src': src_batch, 'trg': trg_batch}


def transformer_collate(batch):
    src_batch, trg_batch = [], []
    
    for src, trg in batch:
        src_batch.append(torch.LongTensor(src))
        trg_batch.append(torch.LongTensor(trg))
    
    batch_size = len(src_batch)
    eos_batch = torch.LongTensor([3 for _ in range(batch_size)])
    
    src_batch = pad_sequence(src_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    trg_batch = pad_sequence(trg_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    trg_batch = torch.column_stack((trg_batch, eos_batch))

    return {'src': src_batch, 'trg': trg_batch}



def load_dataloader(config, split):
    dataset = Dataset(config, split)
    
    if config.model_name == 'transformer':
        return DataLoader(dataset, 
                          batch_size=config.batch_size, 
                          shuffle=False, 
                          collate_fn=transformer_collate, 
                          num_workers=2)
    
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False, 
                      collate_fn=rnn_collate, 
                      num_workers=2)