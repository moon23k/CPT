import re, os, yaml
from tqdm import tqdm
import sentencepiece as spm




def build_vocab():
    assert os.path.exists(f'configs/vocab.yaml')
    assert os.path.exists(f'data/concat.txt')

    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    opt = f"--input=data/concat.txt\
            --model_prefix=data/tokenizer\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/concat.txt')



def tokenize_data(src_data, trg_data, tokenizer):
    tokenized_data = []
    for src, trg in zip(src_data, trg_data):
        temp_dict = dict()
        
        temp_dict['src'] = tokenizer.EncodeAsIds(src)
        temp_dict['trg'] = tokenizer.EncodeAsIds(trg)
        
        tokenized_data.append(temp_dict)
    
    return tokenized_data



def split_data(dataset, downsize):
    src, trg = [], []
    for dial in tqdm(dataset):
        _seq = dial.split("__eou__")[:-1]
        seq_len = len(_seq)
        seq = []
        
        for uttr in _seq:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            seq.append(_uttr.strip())
        
        if seq_len < 2:
            continue

        elif seq_len == 2:
            src.append(seq[0])
            trg.append(seq[1])
            continue

        #Incase of seq_len is even
        elif seq_len % 2 == 0:
            src.extend(seq[0::2])
            trg.extend(seq[1::2])

            src.extend(seq[1:-1:2])
            trg.extend(seq[2::2])
        
        #Incase of seq_len is odds
        elif seq_len % 2 == 1:
            src.extend(seq[0:-1:2])
            trg.extend(seq[1::2])
            
            src.extend(seq[1::2])
            trg.extend(seq[2::2])   

    assert len(src) == len(trg)

    if downsize:
        src, trg = src[::2], trg[::2]

    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(src + trg))

    return src, trg




def main(orig_data, downsize=True):
    src, trg = split_data(orig_data, downsize)
    build_vocab()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

    tokenized_data = tokenize_data(src, trg, tokenizer)
    
    train, valid, test = tokenized_data[:-6000], tokenized_data[-6000:-3000], tokenized_data[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)



if __name__ == '__main__':
    assert os.path.exists(f'data/dialogues_text.txt')
    with open('data/dialogues_text.txt', 'r') as f:
        orig_data = f.readlines()

    main(orig_data)
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')