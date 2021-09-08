import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import math
import random

from transformers import *
from math import ceil, floor

# Optimizer
class ModelOptimizer(object):
    def __init__(self, transformer_optimizer, transformer_scheduler,
                 task_optimizer, task_init_lr, max_iter):
        self.iter = 0
        self.transformer_optimizer = transformer_optimizer
        self.transformer_scheduler = transformer_scheduler

        self.task_optimizer = task_optimizer
        self.task_init_lr = task_init_lr
        self.max_iter = max_iter

    def zero_grad(self):
        if self.transformer_optimizer:
            self.transformer_optimizer.zero_grad()
        if self.task_optimizer:
            self.task_optimizer.zero_grad()

    def step(self):
        self.iter += 1
        if self.transformer_optimizer:
            self.transformer_optimizer.step()
        if self.task_optimizer:
            self.task_optimizer.step()
            self.poly_lr_scheduler(self.task_optimizer, self.task_init_lr, self.iter, self.max_iter)
        if self.transformer_scheduler:
            self.transformer_scheduler.step()

    @staticmethod
    def poly_lr_scheduler(optimizer, init_lr, iter, max_iter,
                          lr_decay_iter=1, power=1.0):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param max_iter is number of maximum iterations
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param power is a polymomial power
        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr*(1 - iter/max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

# BaseModel
class BaseModel(nn.Module):
    def __init__(self, configs):
        super(BaseModel, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')

    def get_optimizer(self, num_train_examples, use_transformer_scheduler=True):
        # Compute num_warmup_steps and num_train_optimization_steps
        configs = self.configs
        num_epoch_steps = math.ceil(num_train_examples / configs['batch_size'])
        num_train_optimization_steps = \
            int(num_epoch_steps * configs['epochs'] / configs['gradient_accumulation_steps'])
        num_warmup_steps = int(num_train_optimization_steps * 0.1)

        # Extract transformer parameters and task-specific parameters
        transformer_params, task_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('transformer'):
                    transformer_params.append((name, param))
                else:
                    task_params.append((name, param))

        # Prepare transformer_optimizer and transformer_scheduler
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in transformer_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in transformer_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        transformer_optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.configs['transformer_learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-06,
        )
        transformer_scheduler = None
        if use_transformer_scheduler:
            transformer_scheduler = get_linear_schedule_with_warmup(transformer_optimizer,
                                                                    num_warmup_steps=num_warmup_steps,
                                                                    num_training_steps=num_train_optimization_steps)

        # Prepare the optimizer for task-specific parameters
        if len(task_params) > 0:
            task_optimizer = optim.Adam([p for n, p in task_params], lr=self.configs['task_learning_rate'])
        else:
            task_optimizer = None

        # Unify transformer_optimizer and task_optimizer
        model_optimizer = ModelOptimizer(transformer_optimizer, transformer_scheduler,
                                         task_optimizer, self.configs['task_learning_rate'],
                                         num_train_optimization_steps)
        model_optimizer.iter = 0

        return model_optimizer

# FFNN Module
class FFNNModule(nn.Module):
    """ Generic FFNN-based Scoring Module
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout = 0.2):
        super(FFNNModule, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.squeeze()

# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
