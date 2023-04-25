import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration



# BART
class BART(nn.Module):
    def __init__(self, config, tokenizer):
        super(BART, self).__init__()
        self.pretrained = config.pretrained
        self.bart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        if not self.pretrained:
            bart_config = self.bart.config
            self.bart = BartForConditionalGeneration(bart_config)
        self.tokenizer = tokenizer


    def make_mask(self, src):
        mask = torch.where(src==self.tokenizer.pad_token_id, 0, 1)
        return mask
        

    def forward(self, src, trg):
        enc_mask, dec_mask = self.make_mask(src), self.make_mask(trg)
        output = self.bart(input_ids=src, attention_mask=enc_mask, decoder_input_ids=trg, decoder_attention_mask=dec_mask).logits
        return output