import numpy as np
import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, task, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(task, split)


    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['x']).ids
        y = self.tokenizer.encode(self.data[idx]['y']).ids
        return torch.LongTensor(x), torch.LongTensor(y)




class Collator(object):
    def __init__(self, config):
        
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id
        self.mask_id = config.mask_id
        self.vocab_size = config.vocab_size
        self.pretrain = config.mode == 'pretrain'


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        x_batch = self.pad_batch(x_batch)
        y_batch = self.pad_batch(y_batch) if not self.pretrain else x_batch

        if self.pretrain:
            x_batch = self.masking(x_batch)

        return {'x': x_batch, 
                'y': y_batch}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True,
            padding_value=self.pad_id
        )


    def masking(self, x):
        rand = torch.rand(x.shape)
        
        mask_arr = (rand < 0.15) * (x != self.bos_id) * (x != self.eos_id) * (x != self.pad_id)

        selection = torch.flatten((mask_arr[0]).nonzero())
        selection_val = np.random.random(len(selection))

        mask_selection = selection[np.where(selection_val >= 0.2)[0]]  # 80%: Mask Token
        random_selection = selection[np.where(selection_val < 0.1)[0]] # 10%: Random Token

        x[0, mask_selection] = self.mask_id
        x[0, random_selection] = torch.randint(0, self.vocab_size, size=random_selection.shape)

        return x



def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, config.task, split), 
        batch_size=config.batch_size, 
        shuffle=not split=='test',
        collate_fn=Collator(config),
        num_workers=2
    )