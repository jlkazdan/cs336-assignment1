import torch
import json
from typing import Dict, List, Tuple
from functools import lru_cache
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    pre_tokens = list(pre_token_freqs.keys())
    for pre_token in pre_tokens:
        if not is_subsequence(pair, pre_token):
            pass
        else:
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
