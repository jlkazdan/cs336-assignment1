import torch
import numpy as np
import regex as re
import torch.nn as nn
from collections import defaultdict
import ast 
import multiprocessing as mp
from cs336_basics.utils import get_largest_key, update_counts, load_tokenizer_details, find_chunk_boundaries
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Pool  
import time

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
        print('one chunk pre-tokenized')
        return (pre_token_freqs, token_freqs)


    def train_bpe(self, corpus, workers = 1, num_chunks = 1):
        #initialize frequences
        token_freqs = defaultdict(int)
        pre_token_freqs = defaultdict(int)
        start = time.time()
        with open(corpus, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')
            
            # Process each chunk
            chunks = []
            for i in range(len(boundaries)-1):
                f.seek(boundaries[i])
                sub_text = f.read(boundaries[i+1]-boundaries[i]).decode('utf-8')
                chunks.append(sub_text)
            print('got chunks to read')

            end_of_reading = time.time()
            print(f'finished reading in the file: {end_of_reading - start}')
            #Initialize the vocabulary with the original 256 byte pairs
            if num_chunks > 1:
                # with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                #     output_tuples = executor.map(self.pre_tokenize, portions)

                with Pool(workers) as p:
                    output_tuples = p.map(self.pre_tokenize, chunks)
                for ele in output_tuples:
                    pre_token_freqs |= ele[0]
                    token_freqs |= ele[1]
            else:
                pre_token_freqs, token_freqs = self.pre_tokenize(chunks[0])
            end_of_pre_tokenization = time.time()

            print(f'got pre-tokens: {end_of_pre_tokenization - end_of_reading}')

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
            key = max(token_freqs, key = lambda k: (token_freqs[k], k))
            merge_list.append((key[0].encode(), key[1].encode()))
            idx = i+start_vocab_size
            self.vocab[idx] = (key[0]+key[1]).encode() #eventually we will need to change this
            update_counts(pre_token_freqs, token_freqs, key, idx)
        final = time.time()
        print(f'finished training: {final - end_of_pre_tokenization}')
        return self.vocab, merge_list

class Tokenizer(nn.Module):
    def __init__(self, vocab, merges, special_tokens=  None):
        super(Tokenizer, self).__init__()
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = {(self.merges[ind][0], self.merges[ind][1]):ind for ind in range(len(self.merges))}
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
    def from_files(cls, filepath):
        #assume that the file only has a single line
        vocab, merges = load_tokenizer_details(filepath)

        
        return cls(vocab, merges, )

    def encode(self, text:str, workers = 1):
        tokens = []
        if self.special_tokens:
            special_regex = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            chunks = re.split(special_regex, text)
        else:
            chunks = [text]
        
        

        if workers == 1:
            for chunk in chunks:
                tokens += self.encode_chunk(chunk)
        else:
            with Pool(workers) as p:
                token_chunks = p.map(self.encode_chunk, chunks)
            for token_chunk in token_chunks:
                tokens += token_chunk 
        print('tokenization completed')
        return tokens
        

    #decode one chunk of text without any special tokens
    def encode_chunk(self, text:str):
        # Check if the entire text is already in the vocabulary
        if text.encode("utf-8") in self.vocab_to_index:
            return [self.vocab_to_index[text.encode("utf-8")]]
        
        # Tokenize the text using the pattern
        pre_tokens = re.findall(PAT, text)
        
        # Convert each token to bytes
        for i, pretoken in enumerate(pre_tokens):
            enc = pretoken.encode()
            pre_tokens[i] = [enc[i:i+1] for i in range(len(enc))]
        
        encoded = []
        
        # Process each pretoken independently
        for pretoken in pre_tokens:
            # Continue merging until no more merges are possible
            while len(pretoken) > 1:
                potential_merges = []
                # Find all possible merges in the current token list
                for i in range(len(pretoken)-1):
                    pair = (pretoken[i], pretoken[i+1])
                    if pair in self.merges_dict:
                        potential_merges.append((self.merges_dict[pair], pair))
                
                # If no merges are possible, break out
                if len(potential_merges) == 0:
                    break
                    
                # Find the merge with the lowest priority (minimum index)
                next_merge = min(potential_merges)[1]
                
                # Apply the merge to the token list
                revised_pretoken = []
                i = 0
                while i < len(pretoken):
                    # If we find the pair to merge and we're not at the end
                    if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == next_merge:
                        # Merge the pair and add it
                        revised_pretoken.append(pretoken[i] + pretoken[i+1])
                        i += 2
                    else:
                        # Otherwise just add the current token
                        revised_pretoken.append(pretoken[i])
                        i += 1
                
                # Update pretoken for the next iteration
                pretoken = revised_pretoken
            
            # Add tokens to the final encoded list
            for token in pretoken:
                if token in self.vocab_to_index:
                    encoded.append(self.vocab_to_index[token])
                else:
                    # Handle unknown tokens if needed
                    # This part depends on how you want to handle unknown tokens
                    pass
                    
        return encoded
    
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

            



    def decode(self, ids:list[int]):
        to_decode = b''.join([self.vocab[id] for id in ids])
        return to_decode.decode("utf-8", errors = "replace")


