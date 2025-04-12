import torch
import json
from typing import Dict, List, Tuple, BinaryIO
from functools import lru_cache
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import os 

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def update_counts_numeric(pre_token_freqs, token_freqs, pair, idx, pair_to_pretokens):
    first_elem, second_elem = pair
    pre_tokens = list(pair_to_pretokens[pair])
    #pre_tokens = [pt for pt in pre_token_freqs.keys() if first_elem in pt and second_elem in pt]
    for pre_token in pre_tokens:
            count = pre_token_freqs[pre_token]

            new_pre_token = []
            
            i = 0
            while i < len(pre_token):
                if i < len(pre_token) - 1 and pre_token[i] == pair[0] and pre_token[i+1] == pair[1]:
                    new_pre_token.append(idx)
                    #update the frequencies
                    if i > 0:
                        subtract_pair = (pre_token[i-1], pre_token[i])
                        add_pair = (pre_token[i-1], idx)
                        token_freqs[subtract_pair] -= count
                        token_freqs[add_pair] += count
                    if i < len(pre_token)-2:
                        subtract_pair = (pre_token[i+1], pre_token[i+2])
                        add_pair = (idx, pre_token[i+2])
                        token_freqs[subtract_pair] -= count
                        token_freqs[add_pair] += count
                    i+=2
                else:
                    new_pre_token.append(pre_token[i])
                    i+=1
            new_pre_token = tuple(new_pre_token)
            pairs_old = set()
            pairs_new = set()
            for i in range(len(pre_token)-1):
                pairs_old.add((pre_token[i], pre_token[i+1]))
            for i in range(len(new_pre_token)-1):
                pairs_new.add((new_pre_token[i], new_pre_token[i+1]))

            for duo in pairs_old:
                pair_to_pretokens[duo].remove(pre_token)

            for duo in pairs_new:
                pair_to_pretokens[duo].add(new_pre_token)
                


            del pre_token_freqs[pre_token]
            pre_token_freqs[new_pre_token] = count 
    del token_freqs[pair]

#this function was borrowed from gpt2's repo in order to convert from otherwise unsaveable bytes to unicode
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def save_tokenizer_details(path: str, vocab: Dict[int, bytes], merges:List[Tuple[bytes, bytes]]):
    b2u = bytes_to_unicode()
    vocab = {num: ''.join(b2u[byte] for byte in vocab[num]) for num in vocab.keys()}
    merges = [(''.join([b2u[byte] for byte in ele[0]]), ''.join([b2u[byte] for byte in ele[1]])) for ele in merges]
    data = {'vocab': vocab, 'merges': merges}
    with open(path, 'w', encoding = 'utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

    print(f'Saved to {path}')



def load_tokenizer_details(path: str):
    b2u = bytes_to_unicode()
    u2b = {b2u[value]: value for value in b2u}
    with open(path) as f:
        data = json.load(f)
    
    vocab_raw = data['vocab']
    merges_raw = data['merges']

    def unicode_to_bytes(unicode_string):
        return bytes(u2b[char] for char in unicode_string)
    
    vocab = {num: unicode_to_bytes(vocab_raw[num]) for num in vocab_raw}
    merges = [(unicode_to_bytes(ele[0]), unicode_to_bytes(ele[1])) for ele in merges_raw]
    return vocab, merges
    


def get_largest_key(dictionary):
    assert len(dictionary) != 0, "Dictionary cannot be empty"
    largest = 0
    ret = None
    for key in dictionary:
        if dictionary[key] >= largest:
            if dictionary[key] > largest or ret is None or key > ret:
                ret = key
                largest = dictionary[key]
    return ret


def is_subsequence(smaller, larger):
    # Check if smaller tuple appears as an ordered subsequence in larger tuple
    n, m = len(smaller), len(larger)
    
    # If smaller is longer than larger, it can't be a subsequence
    if n > m:
        return False
        
    # Check for the subsequence at each possible starting position
    for i in range(m - n + 1):
        if larger[i:i+n] == smaller:
            return True
            
    return False

def update_counts(pre_token_freqs, token_freqs, pair, idx):
    first_elem, second_elem = pair
    pre_tokens = [pt for pt in pre_token_freqs.keys() if first_elem in pt and second_elem in pt]
    for pre_token in pre_tokens:
            count = pre_token_freqs[pre_token]

            new_pre_token = []
            
            i = 0
            while i < len(pre_token):
                if i < len(pre_token) - 1 and pre_token[i] == pair[0] and pre_token[i+1] == pair[1]:
                    replacement_str = pair[0]+pair[1]
                    new_pre_token.append(replacement_str)
                    #update the frequencies
                    if i > 0:
                        subtract_pair = (pre_token[i-1], pre_token[i])
                        add_pair = (pre_token[i-1], replacement_str)
                        token_freqs[subtract_pair] -= count
                        token_freqs[add_pair] += count
                    if i < len(pre_token)-2:
                        subtract_pair = (pre_token[i+1], pre_token[i+2])
                        add_pair = (replacement_str, pre_token[i+2])
                        token_freqs[subtract_pair] -= count
                        token_freqs[add_pair] += count
                    i+=2
                else:
                    new_pre_token.append(pre_token[i])
                    i+=1
            new_pre_token = tuple(new_pre_token)
            del pre_token_freqs[pre_token]
            pre_token_freqs[new_pre_token] = count 
    del token_freqs[pair]


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
