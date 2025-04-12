import torch
import numpy as np
import regex as re
import torch.nn as nn
from collections import defaultdict, Counter
import ast 
import multiprocessing as mp
from cs336_basics.utils import get_largest_key, update_counts_numeric, load_tokenizer_details, find_chunk_boundaries
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Pool  
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETrainer:
    def __init__(self, vocab_size, special_tokens):
        assert vocab_size > 256, "Must have a vocab size greater than or equal to 256"
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        for byte_value in range(256):
            self.vocab[byte_value] = bytes([byte_value])
        for token in self.special_tokens:
            self.vocab[len(self.vocab)]= token.encode("utf-8")
        self.pairs_to_pretokens = {}

    def process_chunk(self, data):
        corpus, start, end = data
        with open(corpus, 'rb') as f:
            f.seek(start)
            sub_text = f.read(end-start).decode('utf-8')
        if self.special_tokens:
            special_regex = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            splits = re.split(special_regex, sub_text)
        else:
            splits = [sub_text]
        
        #now go through and process each of the sub splits
        pre_token_freqs = defaultdict(int)
        for ele in splits:
            if ele in self.special_tokens:
                pass #we don't need to consider the special tokens
            else:
                pre_tokens = re.finditer(PAT, ele)
                for gr in pre_tokens:
                    pre_token_freqs[tuple(gr.group(0).encode('utf8'))] += 1

        pair_freqs = defaultdict(int)
        pair_to_pretokens = defaultdict(set)
        #now get pairs to frequencies
        for pre_token in pre_token_freqs:
            for i in range(len(pre_token)-1):
                pair = (pre_token[i], pre_token[i+1])
                pair_freqs[pair] += pre_token_freqs[pre_token]
                pair_to_pretokens[pair].add(pre_token)
        return (pre_token_freqs, pair_freqs, pair_to_pretokens)

    def train(self, corpus, num_chunks=10, num_workers=5):
        pre_token_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        pair_to_pretokens = defaultdict(set)
        #chunk the document
        with open(corpus, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')
        
        chunks = []
        for i in range(len(boundaries)-1):
            chunks.append((corpus, boundaries[i], boundaries[i+1]))
        output_tuples = []
        # for i in range(num_workers):
        #     output_tuples.append(self.process_chunk(chunks[i]))
        with Pool(num_workers) as p:
            output_tuples = p.map(self.process_chunk, chunks)
        for output_tuple in output_tuples:
            temp_pre_tok_freqs = output_tuple[0]
            for ele in temp_pre_tok_freqs:
                pre_token_freqs[ele] += temp_pre_tok_freqs[ele]
            pair_freqs |= output_tuple[1]
            pair_to_pretokens_partial = output_tuple[2]
            for ele in pair_to_pretokens_partial:
                pair_to_pretokens[ele] = pair_to_pretokens[ele].union(pair_to_pretokens_partial[ele])
        

        print('finished read in...')

        merges = self.vocab_size - len(self.vocab)
        start_vocab_size = len(self.vocab)
        merge_list = []
        for i in range(merges):
            #if (i+1)%100 == 0:
            #    print(i+1)
            print(i+1)

            key = max(pair_freqs, key = lambda k: (pair_freqs[k], (self.vocab[k[0]], self.vocab[k[1]])))
            merge_list.append((self.vocab[key[0]], self.vocab[key[1]]))
            idx = i+start_vocab_size

            self.vocab[idx] = self.vocab[key[0]]+self.vocab[key[1]] #eventually we will need to change this
            #print(f'before {max(pair_freqs, key = lambda k: (pair_freqs[k]))}')
            update_counts_numeric(pre_token_freqs, pair_freqs, key, idx, pair_to_pretokens)
            #print(f'after {max(pair_freqs, key = lambda k: (pair_freqs[k]))}')
        return self.vocab, merge_list