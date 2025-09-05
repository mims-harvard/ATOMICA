#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import random
import os


def setup_seed(seed):
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True


SEED = 12