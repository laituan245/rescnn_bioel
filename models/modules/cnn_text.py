# Adapted from the paper https://arxiv.org/pdf/1408.5882.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import FFNNModule
from torch.autograd import Variable
from models.modules.self_attention import SelfAttention

class CNN_Block(nn.Module):
    def __init__(
            self,
            hidden_dim=256,
            n_filters=100,
            output_dim=256,
            kernel_sizes=[1,3,5],
            paddings=[0,1,2]
        ):
        super(CNN_Block, self).__init__()
        input_dim = output_dim = hidden_dim

        # convs_module
        self.convs_module = []
        for kernel_size, padding in zip(kernel_sizes, paddings):
            self.conv = nn.Conv1d(input_dim, n_filters, kernel_size, stride=1, padding=padding)
            self.convs_module.append(self.conv)
        self.convs_module = nn.ModuleList(self.convs_module)

        # layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)

        # FFNN module
        ffnn_input_dim = len(kernel_sizes) * n_filters
        ffnn_hidden_dim = (ffnn_input_dim + output_dim) // 2
        self.ffnn = FFNNModule(ffnn_input_dim,
                               [ffnn_hidden_dim],
                               output_dim)

    def forward(self, input):
        residual = input
        batch_size, feature_size, seq_len = residual.size()

        outputs = []
        for conv in self.convs_module:
            outputs.append(torch.relu(conv(input)))
        outputs = torch.cat(outputs, dim=1)

        # Apply FFNN
        outputs = self.ffnn(torch.transpose(outputs, 1, 2))
        if seq_len == 1 and len(outputs.size()) == 2:
            outputs = outputs.unsqueeze(1)
        outputs = torch.transpose(outputs, 1, 2)

        return residual + outputs


class CNN_Text(nn.Module):
    def __init__(
            self,
            input_dim=768,
            pooling_type='max',
            hidden_dim=256,
            n_filters=100,
            kernel_sizes=[1,3,5],
            paddings=[0,1,2],
            depth = 12,
            dropout = 0.2
        ):
        super(CNN_Text, self).__init__()
        layers = []

        # Project embedding_dim into base_num_features
        layers.append(nn.Conv1d(input_dim, hidden_dim, 1))

        # CNN_Block
        for _ in range(depth):
            cnn_block = CNN_Block(
                hidden_dim=hidden_dim,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                paddings=paddings
            )
            layers.append(cnn_block)

        self.layers = nn.Sequential(*layers)

        # Pooling
        assert(pooling_type in ['max', 'attention', 'mean'])
        self.pooling_type = pooling_type
        if self.pooling_type == 'attention':
            self.self_attention = SelfAttention(hidden_dim)

    def forward(self, x, attention_mask):
        output = x.transpose(1, 2)
        output = self.layers(output)

        if self.pooling_type == 'max':
            final_output = torch.max(output, dim=-1)[0]
        elif self.pooling_type == 'mean':
            final_output = torch.mean(output, dim=-1)
        elif self.pooling_type == 'attention':
            final_output = self.self_attention(output, attention_mask)
        else:
            raise NotImplementedError
        return final_output
