import torch

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import Optional, Callable

import numpy.typing as npt
from torch import Tensor
import numpy as np

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super(AdamW, self).__init__(params, defaults)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.lamb = weight_decay


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
    
                grad = p.grad.data
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)
                if 'v' not in state:
                    state['v'] = torch.zeros_like(p.data)
                if 't' not in state:
                    state['t'] = 1
                t = state['t']
                state['m'] = self.beta1*state['m'] + (1-self.beta1)*grad
                state['v'] = self.beta2*state['v'] + (1-self.beta2)*grad**2
                alpha_t = lr * np.sqrt(1-self.beta2 ** t)/(1-self.beta1**t)
                p.data -= alpha_t * state['m']/(np.sqrt(state['v'] + self.eps))
                p.data -= lr*self.lamb*p.data
                state['t'] += 1
        
    def change_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha_t = t/T_w*alpha_max
    elif T_w <= t and t<= T_c:
        alpha_t = alpha_min + 0.5*(1 + np.cos((t-T_w)/(T_c- T_w)*np.pi))*(alpha_max - alpha_min)
    else:
        alpha_t = alpha_min

    return alpha_t

def gradient_clipping(params, M, epsilon=1e-6):
    magnitude = 0
    for p in params:
        if p.grad is not None:  # Check if gradient exists
            magnitude += torch.sum(p.grad.data**2)  # Use .data for efficiency
    
    magnitude = magnitude**0.5  # Calculate L2 norm
    
    if magnitude > M:
        scalar = M / (magnitude + epsilon)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scalar)  # In-place multiplication
    
