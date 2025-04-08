import torch

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import Optional, Callable

import numpy.typing as npt
import torch
from torch import Tensor
import numpy as np

def data_loading(token_ids, batch_size, context_length, device = 'cpu'):
    top_index = len(token_ids) - context_length
    start_indices = np.random.randint(0, top_index, size=(batch_size, ))
    end_indices = start_indices + context_length
    
    batches = []
    targets = []
    for start, end in zip(start_indices, end_indices):
        batches.append(token_ids[start:end])
        targets.append(token_ids[start+1:end+1])
    X = np.array(batches)
    X = torch.from_numpy(X).to(device)
    targets = torch.from_numpy(np.array(targets)).to(device)

    return (X, targets)