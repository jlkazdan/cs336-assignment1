import cProfile
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.train_bpe import BPETrainer
from cs336_basics.utils import save_tokenizer_details
import time

path = "/data/TinyStoriesV2-GPT4-train.txt"
save_path = "cs336_basics/output_tokenizers/TinyStories-train.json"
vocab_size = 10**4
special_tokens = ["<|endoftext|>"]

def main():
    profile = cProfile.Profile()
    profile.enable()
    start = time.time()
    tokenizer = BPETrainer(vocab_size, special_tokens)
    vocab, merges = tokenizer.train(path, num_workers = 8, num_chunks=16)
    save_tokenizer_details(save_path, vocab, merges)
    tokens = [vocab[key] for key in vocab]
    end = time.time()
    print(f'the longest token is {max(tokens, key=len)}')
    profile.disable()
    profile.print_stats(sort = 'time')
    print(f'the total time is {end-start}')
if __name__ == "__main__":
    main()
