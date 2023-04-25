from transformers import PreTrainedTokenizerFast



class BARTTokenizer:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.convert_ids_to_tokens(0), self.tokenizer.convert_tokens_to_ids('<s>')
        self.sep_token, self.sep_token_id = self.tokenizer.convert_ids_to_tokens(1), self.tokenizer.convert_tokens_to_ids('</s>')
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        return self.tokenizer.encode(s)


    def decode(self, tok):
        try:
            tok = tok[:tok.index(self.sep_token_id)]
        except ValueError:
            try:
                tok = tok[:tok.index(self.pad_token_id)]
            except:
                pass
        return self.tokenizer.decode(tok)