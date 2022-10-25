import torch
import sentencepiece as spm
from run import load_tokenizer
from modules.search import SeqSearch, AttnSearch, TransSearch



class Translator:
	def __init__(self, config, model):
		self.model = model
		self.device = config.device
		self.search = config.search
		self.bos_idx = config.bos_idx
		self.model_name == config.model_name
        self.src_tokenizer = load_tokenizer('src')
		self.trg_tokenizer = load_tokenizer('trg')

        if self.model.training:
            self.model.eval()

        if self.model_name == 'seq2seq':
            self.beam = SeqSearch(config, self.model)
        elif self.model_name == 'attention':
            self.beam = AttnSearch(config, self.model)
        elif self.model_name == 'transformer':
            self.beam = TransSearch(config, self.model)            

	
	def tranaslate(self, config):
		print('Type "quit" to terminate Chat')
		while True:
			user_input = input('Please Type User Text >> ')
			if user_input.lower() == 'quit':
				print('--- Terminate the Chat ---')
				print('-' * 30)
				break

			src = self.src_tokenizer.Encode(user_input)
			src = torch.LongTensor(src).unsqueeze(0).to(self.device)

            if self.search == 'beam':
                pred_seq = self.search.beam_search(src)
            elif self.search == 'greedy':
                pred_seq = self.search.greedy_search(src)

			print(f"User: {user_input}")
			print(f'Bot : {self.trg_tokenizer.Decode(pred_seq)}\n')
			
