import torch, random
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim, 
                           config.hidden_dim, 
                           config.n_layers, 
                           batch_first=True, 
                           dropout=config.dropout_ratio)
        self.dropout = nn.Dropout(config.dropout_ratio)
    
    def forward(self, x):
        x = self.dropout(self.embedding(x)) 
        _, hiddens = self.rnn(x)
        return hiddens


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim,
                           config.hidden_dim, 
                           config.n_layers,
                           batch_first=True,
                           dropout=config.dropout_ratio)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
    
    def forward(self, x, hiddens):
        x = self.dropout(self.embedding(x.unsqueeze(1)))
        out, hiddens = self.rnn(x, hiddens)
        out = self.fc_out(out.squeeze(1))
        return out, hiddens


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.device = config.device
        self.output_dim = config.output_dim
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.zeros(max_len, batch_size, self.output_dim).to(self.device)

        dec_input = trg[:, 0]
        hiddens = self.encoder(src)

        for t in range(1, max_len):
            out, hiddens = self.decoder(dec_input, hiddens)
            outputs[t] = out
            pred = out.argmax(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else pred

        return outputs.contiguous().permute(1, 0, 2)[:, 1:]