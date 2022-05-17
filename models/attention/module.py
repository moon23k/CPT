import random
import torch
import torch.nn as nn
import torch.nn.functional as F




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.emb = nn.Embedding(config.input_dim, config.emb_dim)
        self.rnn = nn.GRU(config.emb_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, src):
        embedded = self.emb(src)
        embedded = self.dropout(embedded)

        out, hidden = self.rnn(embedded)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)

        return out, hidden




class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.attn = nn.Linear((config.hidden_dim * 3), config.hidden_dim)
        self.v = nn.Linear(config.hidden_dim, 1, bias=False)


    def forward(self, hidden, enc_out):
        batch_size, src_len = enc_out.shape[0], enc_out.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        attn_value = self.v(energy).squeeze(2)
        attn_value = F.softmax(attn_value, dim=1)

        #attn_value: [batch_size, seq_len]
        return attn_value





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.output_dim = config.output_dim

        self.attention = Attention(config)

        self.emb = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.GRU((config.hidden_dim * 2) + config.emb_dim, config.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear((config.hidden_dim * 3) + config.emb_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    
    def transform_tensor(self, tensor):
        tensor = tensor.permute(1, 0, 2)
        tensor = tensor.squeeze(0)
        return tensor


    def forward(self, input, hidden, enc_out):
        input = input.unsqueeze(1)

        embedded = self.emb(input)
        embedded = self.dropout(embedded)


        attn_value = self.attention(hidden, enc_out)
        attn_value = attn_value.unsqueeze(1)
        weighted = torch.bmm(attn_value, enc_out)


        rnn_input = torch.cat((embedded, weighted), dim=2)


        out, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        
        hidden = hidden.squeeze(0)
        out = self.transform_tensor(out)

        assert(out == hidden).all()

        embedded = self.transform_tensor(embedded)
        weighted = self.transform_tensor(weighted)

        pred = self.fc_out(torch.cat((out, weighted, embedded), dim=1))        

        return pred, hidden.squeeze(0)




        

class Seq2Seq_Attn(nn.Module):
    def __init__(self, config):
        super(Seq2Seq_Attn, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_out, hidden = self.encoder(src)

        #define container var for saving predictions
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)

        #set first input as <bos> token
        input = trg[:, 0]

        #genrerate predictions as time step goes by
        for t in range(1, trg_len):
            out, hidden = self.decoder(input, hidden, enc_out)
            outputs[t] = out

            top1 = out.argmax(1)

            #apply teacher_forcing randomly
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        #outputs: [batch_size, seq_len, output_dim]
        outputs = outputs.permute(1, 0, 2)
        return outputs