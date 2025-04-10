import cProfile
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.utils import save_tokenizer_details

path = "data/owt_train.txt"
save_path = "cs336_basics/output_tokenizers/owt-train.json"
vocab_size = 3*10**4
special_tokens = ["<|endoftext|>"]

def main():
    profile = cProfile.Profile()
    profile.enable()
    tokenizer = BPETokenizer(vocab_size, special_tokens)
    vocab, merges = tokenizer.train_bpe(path, workers = 8, num_chunks=1000)
    save_tokenizer_details(save_path, vocab, merges)
    tokens = [vocab[key] for key in vocab]
    print(f'the longest token is {max(tokens, key=len)}')
    profile.disable()
    profile.print_stats(sort = 'time')

if __name__ == "__main__":
    main()