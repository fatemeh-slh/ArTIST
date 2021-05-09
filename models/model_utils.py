from __future__ import print_function
import torch
from torch import nn
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
        return torch.autograd.Variable(x, volatile=volatile)
