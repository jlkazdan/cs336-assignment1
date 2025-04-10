import torch
import numpy as np

from cs336_basics.architectures import *
from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.bpe_tokenizer import *
from cs336_basics.data import *
from cs336_basics.utils import *
from cs336_basics.configs.llm_train_config import llm_train_config

def compute_validation_loss(model, data, config):
    sentences, targets = data_loading(data, config.batch_size, config.context_length, device=config.device)
    logits = model(sentences)  # Shape: [batch_size, context_length, vocab_size]
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    loss = cross_entropy(logits, targets)
    print(f'valid loss: {loss}')


    

def train_loop(model, data, valid_data, config):
    optimizer = AdamW(model.parameters(), lr=10, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps)
    for t in range(config.steps):
        # Update learning rate according to schedule
        lr = learning_rate_schedule(t, config.lr_max, config.lr_min, config.warmup_steps, config.terminal_steps_start)
        optimizer.change_lr(lr)
            
        # Get batch of data
        sentences, targets = data_loading(data, config.batch_size, config.context_length, device=config.device)
        sentences = torch.tensor(sentences, dtype=torch.long, device=config.device)
        targets = torch.tensor(targets, dtype=torch.long, device=config.device)  # Use actual targets
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(sentences)  # Shape: [batch_size, context_length, vocab_size]
        
        # Reshape for loss calculation
        # Reshape logits to [batch_size * context_length, vocab_size]
        logits = logits.view(-1, logits.size(-1))
        # Reshape targets to [batch_size * context_length]
        targets = targets.view(-1)
        
        # Calculate loss
        loss = cross_entropy(logits, targets)
        
        # Backward and optimize
        loss.backward()
        gradient_clipping(model.parameters(), M=config.gradient_norm)
        optimizer.step()
        
        print(f"Step {t}, Loss: {loss.item()}")
        if t%config.save_every == 0:
            checkpoint_path = config.save_location + f'checkpoint-{t}.pt'
            save_checkpoint(model, optimizer, t, checkpoint_path)
        if t%config.validation_every == 0:
            validation_loss = compute_validation_loss(model, valid_data, config)
            print(f"Step {t}, Test Loss: {validation_loss}")
    if t%config.save_every == 0:
        checkpoint_path = config.save_location + f'checkpoint-final.pt'
        save_checkpoint(model, optimizer, t, checkpoint_path)
        


def main():
    config = llm_train_config()
    model = transformer_lm(config.vocab_size, config.context_length, config.num_layers, config.d_model, config.num_heads, config.d_ff, config.context_length, config.rope_theta)
    train_data = np.memmap(config.training_data, dtype = np.uint16, mode = 'r') #can we make this faster by changing the shape?)
    valid_data = np.memmap(config.validation_data, dtype = np.uint16, mode = 'r') #can we make this faster by changing the shape?)
    train_loop(model, train_data, valid_data, config)
        
if __name__ == '__main__':
    main()
    
    
    