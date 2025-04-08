import torch
import torch.nn as nn
from einops import rearrange, einsum
from torch.nn.parameter import Parameter
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights = None, device=None, dtype=None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        std = (2/(in_features + out_features))**0.5
        truncation = 3*std
        
        if weights is None:
            fill = torch.empty(out_features, in_features)
        else:
            fill = weights
        self.linear = Parameter(fill)
        if weights is None:
            nn.init.trunc_normal_(self.linear, std = std, a=-truncation, b=truncation)

        if device is not None:
            self.linear = self.linear.to(device)

    def forward(self, x:torch.Tensor):
        return torch.matmul(x, self.linear.T)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights=  None, device=None, dtype=None):
        super(Embedding, self).__init__()


        std = (2/(num_embeddings + embedding_dim))**0.5
        truncation = 3*std
        
        if weights is None:
            fill = torch.empty(num_embeddings, embedding_dim, dtype = dtype)
        else:
            fill = weights
        self.weights = Parameter(fill)
        if weights is None:
            nn.init.trunc_normal_(self.weights, std = std, a=-truncation, b=truncation)

        if device is not None:
            self.weights = self.weights.to(device)
    
    def forward(self, token_ids):
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, weights= None, device=None, dtype= None):
        super(RMSNorm, self).__init__()
        #they didn't say how one should initialize the learnable gains
        self.eps = eps
        std = (2/d_model)**0.5
        truncation = 3*std
        
        if weights is not None:
            self.learnable_gains = weights
        else:
            self.learnable_gains = torch.empty(d_model, dtype=dtype)
        self.learnable_gains = Parameter(self.learnable_gains)

        if device is not None:
            self.learnable_gains = self.learnable_gains.to(device)
        if weights is None:
            nn.init.trunc_normal_(self.learnable_gains, std= std, a = -truncation, b = truncation)
        
    def forward(self, in_features:torch.Tensor) -> torch.Tensor:
        numerators = in_features * self.learnable_gains#, "batch_size sequence_length d_model, d_model -> batch_size, sequence_length")
        rms = (torch.mean(in_features**2, -1) + self.eps)**0.5
        return numerators / rms.unsqueeze(-1)
        

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, w1_weight =  None, w2_weight= None, w3_weight = None, device=None):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            self.d_ff = ((8/3 * d_model)//64)*64
        else:
            self.d_ff = d_ff

        self.L1 = Linear(d_model, self.d_ff, weights = w1_weight, device= device)
        self.L2 = Linear(self.d_ff, d_model, weights = w2_weight, device = device)
        self.L3 = Linear(d_model, self.d_ff, weights= w3_weight, device = device)

    def forward(self, x):
        x_1 = self.L1(x)
        x_3 = self.L3(x)

        swig_x_1 = x_1/(1+torch.exp(-x_1))

        return self.L2(swig_x_1 * x_3)

class RoPE(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len: int, device= None):
        super(RoPE, self).__init__()
        self.theta = theta
        positions = torch.arange(max_seq_len)
        dimensions = 1/theta**(2*torch.arange(d_k//2)/d_k)
        self.thetas = torch.outer(positions, dimensions)
        self.register_buffer("sin", torch.sin(self.thetas), persistent = False)
        self.register_buffer("cos", torch.cos(self.thetas), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        sin_sliced = self.sin[..., token_positions, :]
        cos_sliced = self.cos[..., token_positions, :]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        odds = cos_sliced * x_even - sin_sliced*x_odd
        evens = sin_sliced * x_even + cos_sliced * x_odd
        stacked = torch.stack((odds, evens), -2)
        stacked_trans = rearrange(stacked, "... a b -> ... b a")
        return rearrange(stacked_trans, "... cardinality emb_dim -> ... (cardinality emb_dim)")
    
class multihead_self_attention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        q_proj_weight = None,
        k_proj_weight = None,
        v_proj_weight = None,
        o_proj_weight = None,
        device = None,
        rope = False,
        theta = None,
        max_seq_len = None,
    ):
        super(multihead_self_attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"
        self.d_k = d_model//num_heads
        self.q_proj = Linear(d_model, d_model, weights=q_proj_weight, device=device)
        self.k_proj = Linear(d_model, d_model, weights=k_proj_weight, device = device)
        self.v_proj = Linear(d_model, d_model, weights=v_proj_weight, device = device)
        self.o_proj = Linear(d_model, d_model, weights=o_proj_weight, device = device)
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        if rope:
            assert theta is not None, "you must define theta"
            assert max_seq_len is not None, "you must define theta"
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        



    def forward(self, input, token_positions = None):
        #compute the projections of the inputs
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(input.shape[-2])
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        

        T= input.shape[1]
        mask = torch.triu(torch.ones(T, T).bool(), diagonal = 1).logical_not()
        attention = scaled_dot_product_attention(q, k, v, mask = mask)
        attention = attention.transpose(1,2)
        attention = attention.contiguous().view(input.shape)
        attention = self.o_proj(attention)
        return attention 

# class multihead_self_attention_with_rope(nn.Module):
#     def __init__(self,
#         d_model: int,
#         num_heads: int,
#         max_seq_len: int,
#         theta: float,
#         q_proj_weight: Float[Tensor, " d_k d_in"] = None,
#         k_proj_weight: Float[Tensor, " d_k d_in"] = None,
#         v_proj_weight: Float[Tensor, " d_v d_in"] = None,
#         o_proj_weight: Float[Tensor, " d_model d_v"] = None,
#     ):
#         super(multihead_self_attention_with_rope, self).__init__()
#         self.d_k = d_model//num_heads
#         self.mha = multihead_self_attention(d_model=d_model, num_heads=num_heads, q_proj_weight=q_proj_weight, 
#                                             k_proj_weight = k_proj_weight, v_proj_weight=v_proj_weight, o_proj_weight=o_proj_weight,
#                                             )
#         self.rope = RoPE(theta, self.d_k, max_seq_len)

#     def forward(self, input, token_positions):

        


class transformer_block(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        super(transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_ff = d_ff

        self.n1 = RMSNorm(d_model)
        self.mha = multihead_self_attention(d_model, num_heads, rope=True, theta = theta, max_seq_len = max_seq_len)
        self.n2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        #create your own token positions for the block

    def forward(self, input):
        x = self.n1(input)
        x = self.mha(x) + input
        out = self.n2(x)
        out = self.ff(out)
        return x + out

    def load_from_dict(self, weights):
        self.n1 = RMSNorm(self.d_model, weights=weights['ln1.weight'])
        self.mha = multihead_self_attention(self.d_model, self.num_heads, rope=True, theta=self.theta, max_seq_len=self.max_seq_len, k_proj_weight=weights['attn.k_proj.weight'], q_proj_weight = weights['attn.q_proj.weight'], v_proj_weight= weights['attn.v_proj.weight'], o_proj_weight=weights['attn.output_proj.weight'])
        self.n2 = RMSNorm(self.d_model, weights=weights['ln2.weight'])
        self.ff = SwiGLU(self.d_model, self.d_ff, w1_weight=weights['ffn.w1.weight'], w2_weight=weights['ffn.w2.weight'],
                        w3_weight = weights['ffn.w3.weight'])
        
                         

        
    


        

    

def softmax(x, dim):
    largest, indices = x.max(dim)
    exps = torch.exp(x- largest.unsqueeze(dim))
    sums = exps.sum(dim)
    return exps/sums.unsqueeze(dim)

def scaled_dot_product_attention(    
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    Q_TK = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")/Q.shape[-1]**0.5
    if mask is not None:
        Q_TK = Q_TK.masked_fill_(mask.logical_not(), -float('inf'))
    sm = softmax(Q_TK, -1)
    return einsum(sm, V, "... n m, ... m d_v -> ... n d_v")

def cross_entropy(predicted_logits, targets):
    batch_indices = torch.arange(predicted_logits.shape[0])
    largest, indices = predicted_logits.max(dim = -1, keepdim = True)
    losses = -predicted_logits + largest + torch.logsumexp(predicted_logits - largest, dim = -1, keepdim=True)
    return losses[batch_indices, targets].mean()

    
