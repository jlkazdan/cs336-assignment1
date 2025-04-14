import torch
# from cs336_basics.utils import load_checkpoint
# from cs336_basics.model import load_model
from cs336_basics.architectures import *
from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.bpe_tokenizer import *
from cs336_basics.data import *
from cs336_basics.utils import *
from cs336_basics.configs.llm_train_config import llm_train_config

#this decodes for a single prompt, in practice should decode a batch
def sample(model, tokenizer, prompt, method = 'greedy', p = 0.3, max_len = 256):
    encoded = tokenizer.encode(prompt)
    for i, ele in enumerate(encoded):
        encoded[i] = int(ele)
    
    tokens = torch.tensor(encoded).reshape(1, -1)
    for i in range(max_len-len(encoded)):
        logits = model(tokens)
        if method == "greedy":
            new_tokens = torch.argmax(logits, -1)[..., -1].reshape(-1, 1).to('cpu')
        elif method == "top-p":
            probs = softmax(logits, -1)
            ordered_probs, indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(ordered_probs, -1)
            mask = (cumulative < p)
            mask[..., 0] = True
            mask_new = torch.zeros_like(mask)
            mask_new.scatter_(dim=-1, index=indices, src=mask)
            logits = logits.masked_fill(mask_new, -1000.0)
            probs = softmax(logits, -1)
            new_tokens = torch.multinomial(probs[:, -1, :], num_samples=1)[...,-1].reshape(-1,1).to('cpu')
        else:
            raise NotImplementedError
        tokens = torch.cat([tokens, new_tokens], -1)
    tokens = [str(token) for token in tokens[0].tolist()]
    decoded = tokenizer.decode(tokens)
    return decoded




            

    
    


def main(src, tokenizer_path):
    config = llm_train_config()
    model = transformer_lm(config.vocab_size, config.context_length, config.num_layers, config.d_model, config.num_heads, config.d_ff, config.context_length, config.rope_theta).to(config.device)
    optimizer = AdamW(model.parameters(), lr=10, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps)
    tokenizer = Tokenizer.from_files(tokenizer_path)
    load_checkpoint(src, model, optimizer)
    
    prompt = "Once upon a time there was a pig named Lily who ate too much. "
    print(sample(model, tokenizer, prompt, method = 'top-p', p = 0.15))
    # p = 10
    # #some starter prefixes
    # text = [" The", "Once", "Hello", "A"]
    # tokenized = []
    # for ele in text:
    #     encoded = tokenizer.encode(ele)
    #     for i, ele in enumerate(encoded):
    #         encoded[i] = int(ele)
    #     tokenized.append(encoded)
    # print(tokenized)
    # tokenized = torch.tensor(tokenized, dtype = int)
    # batch_size = len(text)
    # tokens = torch.tensor(tokenized).reshape(batch_size, 1)

    # for i in range(config.context_length):
    #     logits = model(tokens)
    #     # probs = softmax(logits, -1)
    #     # probs[:p]
    #     new_tokens = torch.argmax(logits, -1)[..., -1].reshape(-1, 1).to('cpu')
    #     tokens = torch.cat([tokens, new_tokens], -1)
    # print(tokens[0])
    # tokens = [str(token) for token in tokens[2].tolist()]
    # decoded = tokenizer.decode(tokens)
    # print(decoded)
        
if __name__ == '__main__':
    tokenizer_path = 'cs336_basics/output_tokenizers/owt-train.json'#TinyStories-train.json'
    src = 'cs336_basics/checkpoints/owt/-0.001-checkpoint-15000.pt' #'cs336_basics/checkpoints/tiny_stories/checkpoint-35000.pt'
    main(src, tokenizer_path)
