import os, re, json, yaml
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




#NMT
def process_translation_data(data_volumn=101100):
    #load original dataset
    nmt_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    
    min_len = 10 
    max_len = 300
    max_diff = 50
    volumn_cnt = 0

    corpus, processed = [], []
    
    for elem in nmt_data:
        temp_dict = dict()
        x, y = elem['en'].strip().lower(), elem['de'].strip().lower()
        x_len, y_len = len(x), len(y)

        #Filtering Conditions
        min_condition = (x_len >= min_len) & (y_len >= min_len)
        max_condition = (x_len <= max_len) & (y_len <= max_len)
        dif_condition = abs(x_len - y_len) < max_diff

        if max_condition & min_condition & dif_condition:
            corpus.append(x)
            corpus.append(y)
            processed.append({'x': x, 'y':y})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    return processed, corpus



#Dialog
def process_dialogue_data():
    corpus, processed = [], []

    #Load original Datasets
    daily_data = load_dataset('daily_dialog')
    blend_data = load_dataset('blended_skill_talk')


    #Daily-Dialogue Dataset Processing
    x_data, y_data = [], []
    for split in ['train', 'validation', 'test']:
        for dial in daily_data[split]['dialog']:
            dial_list = []
            dial_turns = len(dial)

            if max([len(d) for d in dial]) > 300:
                continue
            
            for uttr in dial:
                _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
                _uttr = re.sub(r'([’])\s+', r'\1', _uttr).strip().lower()
                if len(_uttr) > 300:
                    break
                dial_list.append(_uttr)
            
            if dial_turns < 2:
                continue

            elif dial_turns == 2:
                x_data.append(dial_list[0])
                y_data.append(dial_list[1])
                continue  #To avoid duplicate on below condition

            #Incase of dial_turns is even
            elif dial_turns % 2 == 0:
                x_data.extend(dial_list[0::2])
                y_data.extend(dial_list[1::2])

                x_data.extend(dial_list[1:-1:2])
                y_data.extend(dial_list[2::2])
            
            #Incase of dial_turns is odds
            elif dial_turns % 2 == 1:
                x_data.extend(dial_list[0:-1:2])
                y_data.extend(dial_list[1::2])
                
                x_data.extend(dial_list[1::2])
                y_data.extend(dial_list[2::2])   


    assert len(x_data) == len(y_data)
    for x, y in zip(x_data, y_data):        
        corpus.append(x)
        corpus.append(y)
        processed.append({'x': x, 'y': y})


    #Blend Skill Dataset Processing
    for split in ['train', 'validation', 'test']:
        for elem in blend_data[split]:
            prevs = elem['previous_utterance']

            first_uttr = prevs[0].strip().lower()
            second_uttr = prevs[1].strip().lower()
            third_uttr = elem['free_messages'][0].lower()

            corpus.append(first_uttr)
            corpus.append(second_uttr)
            corpus.append(third_uttr)

            processed.append({'x': first_uttr, 'y': second_uttr})
            processed.append({'x': second_uttr, 'y': third_uttr})
    

    return processed, corpus



#Summarization
def process_summarization_data(data_volumn=101100):    
    volumn_cnt = 0
    corpus, processed = [], []
    min_len, max_len = 500, 3000

    #Load Original Dataset
    cnn_data = load_dataset('cnn_dailymail', '3.0.0')

    for split in ['train', 'validation', 'test']:
        for elem in cnn_data[split]:

            x, y = elem['article'], elem['highlights']

            if min_len < len(x) < max_len:
                if len(y) < min_len:
                    
                    #Lowercase
                    x, y = x.lower(), y.lower()

                    #Remove unnecessary characters in trg sequence
                    y = re.sub(r'\n', ' ', y)                 #remove \n
                    y = re.sub(r"\s([.](?:\s|$))", r'\1', y)  #remove whitespace in front of dot

                    processed.append({'x': x, 'y': y})
                    corpus.append(x)
                    corpus.append(y)

                    #End Condition
                    volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break
    
    return processed, corpus          



def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/{task}/tokenizer.json")



def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')




def main():
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)

    #PreProcess Data
    translation_data, translation_corpus = process_translation_data()
    dialogue_data, dialogue_corpus = process_dialogue_data()
    summarization_data, summarization_corpus = process_summarization_data()


    #Train Tokenizer
    corpus = translation_corpus + dialogue_corpus + summarization_corpus
    with open('data/corpus.txt', 'w') as f:
        json.dump(f, corpus)
    train_tokenizer()

    #Save Data
    save_data('translation', translation_data)
    save_data('dialogue', dialogue_data)
    save_data('summarization', summarization_data)



if __name__ == '__main__': 
    main()    