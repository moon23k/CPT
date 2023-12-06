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

        self.pt_obj = config.pt_obj
        self.mlm_prob = config.mlm_prob
        self.is_pretrain = config.mode == 'pretrain'        
        

    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        x_batch = self.pad_batch(x_batch)
        y_batch = self.pad_batch(y_batch)

        if not self.is_pretrain:
            return {'x': x_batch, 'y': y_batch}

        if self.pt_obj == 'casual':
            return {'x': x_batch, 'y': x_batch}

        masked_x, masked_y = self.masking(x_batch)

        if self.pt_obj == 'masked':
            return {'x': masked_x, 'y': masked_y}
        
        return {'x': masked_x, 'y': x_batch}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True,
            padding_value=self.pad_id
        )


    def masking(self, inputs):
        x = inputs.clone()
        y = inputs.clone()
        shape = inputs.shape
        

        prob_matrix = torch.full(shape, self.mlm_prob)

        #Exclude Special Tokens
        special_tokens_mask = torch.ones((x.shape), dtype=torch.bool) \
                            & (x != self.bos_id) \
                            & (x != self.eos_id) \
                            & (x != self.pad_id)
        special_tokens_mask = ~special_tokens_mask
        prob_matrix.masked_fill_(special_tokens_mask, value=0.0)


        #masking indices
        masked_indices = torch.bernoulli(prob_matrix).bool()
        y[~masked_indices] = -100


        #[MASK] Token Replacement for 80% of masked_indices
        indices_replaced = torch.bernoulli(torch.full(shape, 0.8)).bool() & masked_indices
        x[indices_replaced] = self.mask_id

        #Random Token Replacement for 10% of masked_indices
        indices_random = torch.bernoulli(torch.full(shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(5, self.vocab_size, shape, dtype=torch.long) #random tokens w/o special tokens
        x[indices_random] = random_tokens[indices_random]

        return x, y



def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, config.task, split), 
        batch_size=config.batch_size, 
        shuffle=not split=='test',
        collate_fn=Collator(config),
        num_workers=2
    )