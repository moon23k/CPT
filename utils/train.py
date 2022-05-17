import math
import torch
import torch.nn as nn





def seq_train(model, dataloader, criterion, optimizer, config):
    model.train()
    epoch_loss = 0
    total_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(config.device), batch[1].to(config.device)
        
        pred = model(src, trg)
        pred = pred[1:].contiguous().view(-1, config.output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(pred, trg)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=config.clip)
        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 1000 == 0:
            print(f"---- Step: {i+1}/{total_len} Train Loss: {loss}")

    return epoch_loss / total_len




def seq_eval(model, dataloader, criterion, config):
    model.eval()
    epoch_loss = 0
    total_len = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].to(config.device), batch[1].to(config.device)

            pred = model(src, trg, teacher_forcing_ratio=0)

            pred_dim = pred.shape[-1]
            pred = pred[1:].contiguous().view(-1, pred_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(pred, trg)
            epoch_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"---- Step: {i+1}/{total_len} Eval Loss: {loss}")


    return epoch_loss / total_len




def trans_train(model, dataloader, criterion, optimizer, config):
    model.train()
    epoch_loss = 0
    total_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(config.device), batch[1].to(config.device)

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:].contiguous().view(-1)

        pred = model(src, trg_input)
        pred = pred.contiguous().view(-1, config.output_dim)

        loss = criterion(pred, trg_y)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=config.clip)
        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 1000 == 0:
            print(f"---- Step: {i+1}/{total_len} Train Loss: {loss}")

    return epoch_loss / total_len




def trans_eval(model, dataloader, criterion, config):
    model.eval()
    epoch_loss = 0
    total_len = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].to(config.device), batch[1].to(config.device)
            
            trg_input = trg[:, :-1]
            trg_y = trg[:, 1:].contiguous().view(-1)

            pred = model(src, trg_input)
            
            pred_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, pred_dim)

            loss = criterion(pred, trg_y)
            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"---- Step: {i+1}/{total_len} Eval Loss: {loss}")

    return epoch_loss / total_len