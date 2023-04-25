import torch
import re
import os
import pickle
import sys
import random
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist



"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
      

def make_dataset_path(base_path):
    return base_path + 'data/processed/data.pkl'


def preprocess(base_path, w_path):
    r_path = base_path + 'data/raw/total_chatlog.txt'
    
    total_qa = []
    with open(r_path, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines):
        # except for the non-qa pairs
        try:
            q, a = line.strip().split('\t')
            total_qa.append((q, a))
        except ValueError:
            continue

    with open(w_path, 'wb') as f:
        pickle.dump(total_qa, f)


def save_checkpoint(file, model, optimizer, scheduler, best_type):
    file = file[:-3] + '_' + best_type + '.pt'
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, file)
    print('model pt file for {} is being saved\n'.format(best_type))


def bleu_score(ref, pred, weights):
    smoothing = SmoothingFunction().method3
    return corpus_bleu(ref, pred, weights, smoothing)


def nist_score(ref, pred, n):
    return corpus_nist(ref, pred, n)


def cal_scores(ref, pred, type, n_gram):
    assert type in ['bleu', 'nist']
    if type == 'bleu':
        wts = tuple([1/n_gram]*n_gram)
        return bleu_score(ref, pred, wts)
    return nist_score(ref, pred, n_gram)


def tensor2list(ref, pred, tokenizer):
    ref, pred = torch.cat(ref, dim=0)[:, 1:], torch.cat(pred, dim=0)[:, :-1]
    ref = [[tokenizer.tokenize(tokenizer.decode(ref[i].tolist()))] for i in range(ref.size(0))]
    pred = [tokenizer.tokenize(tokenizer.decode(pred[i].tolist())) for i in range(pred.size(0))]
    return ref, pred


def print_samples(ref, pred, ids, tokenizer):
    print('-'*50)
    for i in ids:
        r, p = tokenizer.tokenizer.convert_tokens_to_string(ref[i][0]), tokenizer.tokenizer.convert_tokens_to_string(pred[i])
        print('gt  : {}'.format(r))
        print('pred: {}\n'.format(p))
    print('-'*50 + '\n')


def preprocessing_query(queries, tokenizer):
    speaker_dict = {0: tokenizer.spk1_token_id, 1: tokenizer.spk2_token_id}
    segment_dict = {0: 1, 1: 2}

    tok, seg = [], []
    for i, s in enumerate(queries):
        s_tok = [speaker_dict[i%2]] + tokenizer.encode(s) if i != 0 else \
            [tokenizer.cls_token_id] + [speaker_dict[i%2]] + tokenizer.encode(s)
        
        tok += s_tok
        seg += [segment_dict[i%2]] * len(s_tok)
    return tok, seg