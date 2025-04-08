import cProfile
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.utils import save_tokenizer_details, bytes_to_unicode

path = "data/TinyStoriesV2-GPT4-train.txt"
save_path = "cs336_basics/output_tokenizers/TinyStories-train.json"
vocab_size = 1000#**4
special_tokens = ["<|endoftext|>"]

profile = cProfile.Profile()
profile.enable()
tokenizer = BPETokenizer(vocab_size, special_tokens)
vocab, merges = tokenizer.train_bpe(path, workers = 10)
save_tokenizer_details(save_path, vocab, merges)

profile.disable()
profile.print_stats(sort = 'time')