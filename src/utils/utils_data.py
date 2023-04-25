import torch
from torch.utils.data import Dataset

import pickle
import random
from tqdm import tqdm



class DLoader(Dataset):
    def __init__(self, data, config, tokenizer):
        random.seed(999)        
        self.tokenizer = tokenizer
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.q_max_len, self.a_max_len = config.q_max_len, config.a_max_len

        if config.daily:
            with open(config.base_path + 'data/processed/daily.pkl', 'rb') as f:
                daily_data = pickle.load(f)
            data += daily_data

        self.data = self.make_qa_pair(data)
        self.length = len(self.data)    
        
    
    def make_qa_pair(self, data):
        total_qa_token = []

        for d in tqdm(data, desc='loading to memory...'):
            q = [self.cls_token_id] + self.tokenizer.encode(d[0])[:self.q_max_len-2] + [self.sep_token_id]
            a = [self.cls_token_id] + self.tokenizer.encode(d[1])[:self.a_max_len-2] + [self.sep_token_id]
            
            q += [self.pad_token_id] * (self.q_max_len - len(q))
            a += [self.pad_token_id] * (self.a_max_len - len(a))
            
            total_qa_token.append((q, a))    
        
        return total_qa_token


    def __getitem__(self, idx):
        q, a = self.data[idx]
        return torch.LongTensor(q), torch.LongTensor(a)


    def __len__(self):
        return self.length