import torch
import numpy as np
import regex as re
import torch.nn as nn
from collections import defaultdict
import ast 
import multiprocessing as mp
from cs336_basics.utils import get_largest_key, update_counts
import concurrent.futures
#the pretokenization regex 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer(nn.Module):
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        super(BPETokenizer, self).__init__()
        assert vocab_size > 256, "Must have a vocab size of at least 256."
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        for byte_value in range(256):
            self.vocab[byte_value] = bytes([byte_value])
        for token in self.special_tokens:
            self.vocab[len(self.vocab)]= token.encode("utf-8")
        
        
    def pre_tokenize(self, corpus):

        pre_tokens = re.findall(PAT, corpus)
        pre_token_freqs = defaultdict(int)
        token_freqs = defaultdict(int)
        for pre_token in pre_tokens:
            encoded_pre_token =tuple(list(pre_token))
            pre_token_freqs[encoded_pre_token] += 1
            for i in range(len(encoded_pre_token)-1):
                token_freqs[(encoded_pre_token[i], encoded_pre_token[i+1])] += 1

        return (pre_token_freqs, token_freqs)


    def train_bpe(self, corpus, workers = 1):
        #initialize frequences
        token_freqs = defaultdict(int)
        pre_token_freqs = defaultdict(int)
        with open(corpus) as f:
            text = f.read()
            print('finished reading in the file')
            #Initialize the vocabulary with the original 256 byte pairs
            if workers > 1:
                portion_length = len(text)//workers
                portions = [text[i:i+portion_length] for i in range(0, len(text), portion_length)]
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    output_tuples = executor.map(self.pre_tokenize, portions)
                print(output_tuples)
                for ele in output_tuples:
                    pre_token_freqs |= ele[0]
                    token_freqs |= ele[1]
            else:
                pre_token_freqs, token_freqs = self.pre_tokenize(text)

            print('got pre-tokens')

            #compute a list of all of the pre-tokens, which act as sequences for the merges
            # for pre_token in pre_tokens:
            #     encoded_pre_token =tuple(list(pre_token))
            #     pre_token_freqs[encoded_pre_token] += 1
            #     for i in range(len(encoded_pre_token)-1):
            #         token_freqs[(encoded_pre_token[i], encoded_pre_token[i+1])] += 1
        merges = self.vocab_size - len(self.vocab)
        start_vocab_size = len(self.vocab)
        merge_list = []
        for i in range(merges):
            key = get_largest_key(token_freqs)
            merge_list.append((key[0].encode(), key[1].encode()))
            idx = i+start_vocab_size
            self.vocab[idx] = (key[0]+key[1]).encode() #eventually we will need to change this
            update_counts(pre_token_freqs, token_freqs, key, idx)

        return self.vocab, merge_list

class tokenizer(nn.Module):
    def __init__(self, vocab, merges, special_tokens=  None):
        super(tokenizer, self).__init__()
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key = len, reverse=True)
        else:
            self.special_tokens = []
        self.vocab_to_index = {vocab[index]: index for index in vocab}
        #add special tokens if they're not already in the list
        for special_token in self.special_tokens:
            if special_token.encode("utf-8") not in self.vocab_to_index:
                self.vocab_to_index[special_token] = len(self.vocab_to_index)
                self.vocab[len(self.vocab)] = special_token.encode("utf-8")
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        #assume that the file only has a single line
        with open(vocab_filepath) as file:
            for line in file:
                vocab = ast.literal_eval(line)
        
        with open(merges_filepath) as merges:
            for line in merges:
                merges = ast.literal_eval(merges)
        
        return cls(vocab, merges, )

    def encode(self, text:str):
        tokens = []
        if self.special_tokens:
            stok_regex = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            chunks = re.split(stok_regex, text)
        else:
            chunks = [text]
        
        
        for chunk in chunks:
            if chunk.encode("utf-8") in self.vocab_to_index:
                tokens.append(self.vocab_to_index[chunk.encode("utf-8")])
            else:
                tokens += self.encode_chunk(chunk)
        return tokens
        

    #decode one chunk of text without any special tokens
    def encode_chunk(self, text:str):
        #do I need to pretokenize in this case?
        
        #do the late merges first
        #the merges is in order, so we can go back to front in the merge list to tokenize the string
        pre_tokens = re.findall(PAT, text)
        
        for i, pretoken in enumerate(pre_tokens):
            enc = pretoken.encode()
            pre_tokens[i] = [enc[i:i+1] for i in range(len(enc))]
        encoded = []
        for pretoken in pre_tokens:
            for merge in self.merges:
                revised_pretoken = []
                i=0
                while i < len(pretoken)-1:
                    if (pretoken[i], pretoken[i+1]) == merge:
                        revised_pretoken.append(pretoken[i]+pretoken[i+1])
                        i+=2
                        
                    else:
                        revised_pretoken.append(pretoken[i])
                        i+=1
                if i == len(pretoken)-1:
                    revised_pretoken.append(pretoken[i])
                pretoken = revised_pretoken
            for ele in revised_pretoken:
                encoded.append(self.vocab_to_index[ele])
        return encoded

    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

            



    def decode(self, ids:list[int]):
        to_decode = b''.join([self.vocab[id] for id in ids])
        return to_decode.decode("utf-8", errors = "replace")


