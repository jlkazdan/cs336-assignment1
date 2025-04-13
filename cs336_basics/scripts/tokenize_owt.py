from typing import Iterable
import numpy as np
import torch
import os
from cs336_basics.utils import find_chunk_boundaries
from cs336_basics.bpe_tokenizer import Tokenizer
import time
import multiprocessing as mp
from multiprocessing import Pool

train = '/data/a1_basics/owt_train.txt'
valid = '/data/a1_basics/owt_valid.txt'
tokenizer_path = 'cs336_basics/output_tokenizers/owt-train.json'
# train = 'data/TinyStoriesV2-GPT4-train.txt'
# valid = 'data/TinyStoriesV2-GPT4-valid.txt'
# tokenizer_path = 'cs336_basics/output_tokenizers/TinyStories-train.json'

workers = 10
num_chunks = 1000

def main():
    tokenizer = Tokenizer.from_files(tokenizer_path)
    for ele in [train, valid]:
        start = time.time()
        
        # Open in binary mode to find chunk boundaries
        with open(ele, 'rb') as f:
            # Get chunk boundaries using newlines as split token
            boundaries = find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')
            
            # Process each chunk
            all_tokens = []
            chunks = []
            for i in range(len(boundaries)-1):
                # f.seek(boundaries[i])
                # sub_text = f.read(boundaries[i+1]-boundaries[i]).decode('utf-8')
                # chunks.append(sub_text)
                start = boundaries[i]
                end = boundaries[i+1]
                chunks.append((ele, start, end))
            
            with Pool(workers) as p:
                tokenized_chunks = p.map(tokenizer.encode_file, chunks) 

            for tokenized_chunk in tokenized_chunks:
                all_tokens += tokenized_chunk
        
            # for i in range(len(boundaries) - 1):
            #     start = time.time()
            #     f.seek(boundaries[i])
            #     chunk_bytes = f.read(boundaries[i+1] - boundaries[i])
            #     chunk_text = chunk_bytes.decode('utf-8')
            #     chunk_tokens = tokenizer.encode(chunk_text, workers=1)
            #     all_tokens.extend(chunk_tokens)
            #     end = time.time()
            #     print(end-start)
        tokens = all_tokens
        print(tokens[:10])
        path = ele.split('/')[1].split(".")[0]
        to_save = np.memmap(f'data/tokenized_data/{path}.bin', dtype=np.uint16, mode='w+', shape=(len(tokens),))
        to_save[:] = tokens
        to_save.flush()
        end = time.time()
        print(f'The time to encode {ele}: {end - start}')

if __name__ == '__main__':
    main()