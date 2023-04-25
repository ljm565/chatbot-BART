import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

import re
import time
import pickle
import random
from tqdm import tqdm
from transformers import top_k_top_p_filtering

from models.bart import BART
from utils.utils_func import *
from utils.config import Config
from tokenizer import BARTTokenizer
from utils.utils_data import DLoader



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.data_path = self.config.data_path
        self.model_path = self.config.model_path
        if self.mode != 'train':
            self.model_path = self.model_path[:-3] + '_' + self.config.model_type + '.pt'
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.q_max_len = self.config.q_max_len
        self.a_max_len = self.config.a_max_len
        self.result_num = self.config.result_num

        # define tokenizer
        self.tokenizer = BARTTokenizer()
        self.config.vocab_size = self.tokenizer.vocab_size

        # dataloader
        if self.mode != 'chatting':
            torch.manual_seed(999)  # for reproducibility
            self.dataset = DLoader(load_dataset(self.data_path), self.config, self.tokenizer)
            data_size = len(self.dataset)
            train_size = int(data_size * 0.95)
            val_size = int(data_size * 0.03)
            test_size = data_size - train_size - val_size

            self.trainset, self.valset, self.testset = random_split(self.dataset, [train_size, val_size, test_size])
            if self.mode == 'train':
                self.dataset = {'train': self.trainset, 'val': self.valset, 'test': self.testset}
                self.dataloaders = {
                    s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                    for s, d in self.dataset.items()}
            else:
                self.dataset = {'test': self.testset}
                self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = BART(self.config, self.tokenizer).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    
        if self.mode == 'train':
            total_steps = len(self.dataloaders['train']) * 100#self.epochs
            pct_start = 100 / total_steps
            final_div_factor = self.lr / 25 / 1e-7    # OneCycleLR default value is 25
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                self.scheduler.load_state_dict(self.check_point['scheduler'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_loss = float('inf')
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    epoch_loss = self.train(phase, epoch)
                    train_loss_history.append(epoch_loss)
                else:
                    epoch_loss = self.test(phase)
                    bleu2, bleu4, nist2, nist4 = self.inference(phase, self.result_num)
                    if phase == 'val':
                        val_loss_history.append(epoch_loss)
                        val_score_history['bleu2'].append(bleu2)
                        val_score_history['bleu4'].append(bleu4)
                        val_score_history['nist2'].append(nist2)
                        val_score_history['nist4'].append(nist4)

                        # save best model for bleu4
                        if  val_score_history['bleu4'][-1] > best_val_bleu:
                            early_stop = 0
                            best_val_bleu = val_score_history['bleu4'][-1]
                            save_checkpoint(self.model_path, self.model, self.optimizer, self.scheduler, 'bleu4')

                        # save best model for loss
                        early_stop += 1
                        if  epoch_loss < best_val_loss:
                            early_stop = 0
                            best_val_loss = epoch_loss
                            best_epoch = best_epoch_info + epoch + 1
                            save_checkpoint(self.model_path, self.model, self.optimizer, self.scheduler, 'loss')
                            
                            self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'val_score_history': val_score_history}
                            print('Saving the loss related data...')
                            with open(self.config.loss_data_path, 'wb') as f:
                                pickle.dump(self.loss_data, f)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val bleu: {:4f}, best epoch: {:d}\n'.format(best_val_bleu, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_score_history': val_score_history}
        return self.loss_data


    def train(self, phase, epoch):
        self.model.train()
        epoch_loss = 0

        for i, (src, trg) in enumerate(self.dataloaders[phase]):
            self.optimizer.zero_grad()
            batch_size = src.size(0)
            src, trg = src.to(self.device), trg.to(self.device)

            with torch.set_grad_enabled(phase=='train'):
                output = self.model(src, trg)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            epoch_loss += loss.item() * batch_size
           
            if i % 200 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))

        epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)

        print('{} loss: {}\n'.format(phase, epoch_loss))

        return epoch_loss


    def test(self, phase):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase + ' testing..'):
                batch_size = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)

                output = self.model(src, trg)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item() * batch_size

        # calculate epoch loss
        epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
        print('{} loss: {:4f}\n'.format(phase, epoch_loss))

        return epoch_loss


    def inference(self, phase, result_num=3):
        self.model.eval()
        all_trg, all_output = [], []
        
        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase+' inferencing..'):
                src, trg = src.to(self.device), trg.to(self.device)
                all_trg.append(trg.detach().cpu())
            
                decoder_all_output = []
                for j in range(self.a_max_len):
                    if j == 0:
                        trg = trg[:, j].unsqueeze(1)
                        output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
                        
                all_output.append(torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1))

        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.tokenizer)
        bleu2 = cal_scores(all_ref, all_pred, 'bleu', 2)
        bleu4 = cal_scores(all_ref, all_pred, 'bleu', 4)
        nist2 = cal_scores(all_ref, all_pred, 'nist', 2)
        nist4 = cal_scores(all_ref, all_pred, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # print samples
        ids = random.sample(list(range(len(all_pred))), result_num)
        print_samples(all_ref, all_pred, ids, self.tokenizer)

        return bleu2, bleu4, nist2, nist4


    def chatting(self, query, num_result=1):
        phone = re.compile('[0-9]{2,3}[- ]?[0-9]{3,4}[- ]?[0-9]{4}')
        replace_num = '###-####-####'

        with torch.no_grad():
            query = [self.tokenizer.cls_token_id] + self.tokenizer.encode(query)[:self.q_max_len-2] + [self.tokenizer.sep_token_id]
            query = query + [self.tokenizer.pad_token_id] * (self.q_max_len - len(query))
            
            query = torch.LongTensor(query).expand(num_result, -1).to(self.device)
            trg = torch.LongTensor([self.tokenizer.cls_token_id]).expand(num_result, -1).to(self.device)

            for _ in range(self.a_max_len):
                output = self.model(query, trg)
                if self.config.greedy:
                    output = torch.argmax(output[:, -1], dim=-1).unsqueeze(1)
                else:
                    output = output[:, -1] / self.config.temperature
                    output = top_k_top_p_filtering(output, top_k=self.config.topk, top_p=self.config.topp)
                    output = torch.multinomial(torch.softmax(output, dim=-1), num_samples=1)
                
                trg = torch.cat((trg, output), dim=1)

                if num_result == 1 and output[0, 0].item() == self.tokenizer.sep_token_id:
                    break

        trg = [self.tokenizer.decode(s[1:].tolist()) for s in trg.detach().cpu()]

        # phone number filtering
        numbers = [phone.findall(s) for s in trg]
        for i in range(len(trg)):
            for number in numbers[i]:
                trg[i] = trg[i].replace(number, replace_num)
            
        return trg