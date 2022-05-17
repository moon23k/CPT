import random
import torch
import torch.nn as nn




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim, config.hidden_dim, config.n_layers, batch_first=True, dropout=config.dropout_ratio)
        self.dropout = nn.Dropout(config.dropout_ratio)
     

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        _, (hidden, cell) = self.rnn(embedded) #hidden, cell : [num_layers, batch_size, hidden_dim]
        
        return hidden, cell




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.output_dim = config.output_dim
        
        self.embedding = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim, config.hidden_dim, config.n_layers, batch_first=True, dropout=config.dropout_ratio)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)


     
    def forward(self, input, hidden, cell):
        #intput: [batch_size] -> [batch_size, 1]
        input = input.unsqueeze(1)
        
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        out, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        out = self.fc_out(out.squeeze(1)) 

        #out: [batch_size, output_dim]
        #hidden, cell: [num_layers, batch_size, hidden_dim]
        return out, hidden, cell




class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #get context vector from encoder
        hidden, cell = self.encoder(src)

        #define container variable for saving predictions 
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)

        #set first input as <bos> token
        input = trg[:, 0] #[batch_size]

        #genrerate predictions by time steps
        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = out
            top1 = out.argmax(1)

            #apply teacher_forcing randomly
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
        
        #outputs: [batch_size, seq_len, output_dim]
        outputs = outputs.permute(1, 0, 2)
        return outputs