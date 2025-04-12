import torch
from cs336_basics.utils import load_checkpoint
from cs336_basics.model import load_model
from cs336_basics.architectures import *
from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.bpe_tokenizer import *
from cs336_basics.data import *
from cs336_basics.utils import *
from cs336_basics.configs.llm_train_config import llm_train_config

def main(model_checkpoint, src, tokenizer_path):
    config = llm_train_config()
    model = transformer_lm(config.vocab_size, config.context_length, config.num_layers, config.d_model, config.num_heads, config.d_ff, config.context_length, config.rope_theta).to(config.device)
    optimizer = AdamW(model.parameters(), lr=10, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps)
    tokenizer = Tokenizer.from_files(tokenizer_path)
    load_checkpoint(src, model, optimizer)
    
    #some starter prefixes
    text = ["The", "Once", "Hello", "A"]
    tokenized = []
    for ele in text:
        tokenizer.encode(ele)
        tokenized.append(ele)
    
    batch_size = len(text)
    tokens = torch.tensor(tokenized).reshape(batch_size, 1)

    for i in range(config.context_length):
        logits = model(tokens)
        print(logits)