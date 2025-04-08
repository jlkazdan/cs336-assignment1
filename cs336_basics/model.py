import torch.nn as nn
from collections import OrderedDict
from cs336_basics.architectures import *

class transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, max_seq_len, theta):
        super(transformer_lm, self).__init__()
        #vocab_size: the size of the vocabulary necessary for determining the dimensionality of the token embedding matrix
        #context_length: maximum context length for determining the dimensionality of the position embedding matrix
        #num_layers: the number of transformer blocks to use
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.Sequential(OrderedDict([(f"trans_{i}", transformer_block(d_model, num_heads, d_ff, max_seq_len, theta)) for i in range(num_layers)]))
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, input):
        x = self.embedding(input)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
    
    def load_weights_from_dict(self, weights):
        self.embedding = Embedding(self.vocab_size, self.d_model, weights=weights['token_embeddings.weight'])
        transformer_blocks = []
        for i in range(self.num_layers):
            layer_weights = dict()
            layer_weights['ln1.weight'] = weights[f'layers.{i}.ln1.weight']
            layer_weights['attn.k_proj.weight'] = weights[f'layers.{i}.attn.k_proj.weight']
            layer_weights['attn.q_proj.weight'] = weights[f'layers.{i}.attn.q_proj.weight']
            layer_weights['attn.v_proj.weight'] = weights[f'layers.{i}.attn.v_proj.weight']
            layer_weights['attn.output_proj.weight'] = weights[f'layers.{i}.attn.output_proj.weight']
            layer_weights['ln2.weight'] = weights[f'layers.{i}.ln2.weight']
            layer_weights['ffn.w1.weight'] = weights[f'layers.{i}.ffn.w1.weight']
            layer_weights['ffn.w2.weight'] = weights[f'layers.{i}.ffn.w2.weight']
            layer_weights['ffn.w3.weight'] = weights[f'layers.{i}.ffn.w3.weight']
            layer = transformer_block(self.d_model, self.num_heads, self.d_ff, self.max_seq_len, self.theta)
            layer.load_from_dict(layer_weights)
            transformer_blocks.append((f'trans_{i}', layer))
        transformer_blocks = OrderedDict(transformer_blocks)
        self.transformer_blocks = nn.Sequential(transformer_blocks)
        self.norm = RMSNorm(self.d_model, weights=weights['ln_final.weight'])
        self.linear = Linear(self.d_model, self.vocab_size, weights=weights['lm_head.weight'])

