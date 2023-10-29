import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        x = self.data[idx]['x']
        y = self.data[idx]['y']
        
        x = self.tokenizer.encode(x).ids
        y = self.tokenizer.encode(y).ids
        
        return torch.LongTensor(x), torch.LongTensor(y)



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        
        return {'x': self.pad_batch(src_batch), 
                'y': self.pad_batch(trg_batch)}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True,
            padding_value=self.pad_id
        )


def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config.pad_id),
        num_workers=2
    )