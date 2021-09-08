# Adapted from the paper https://arxiv.org/pdf/1606.01781.pdf
import torch.nn as nn
from torch.nn.init import kaiming_normal_

class ConvBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1,
                 stride=1, shortcut=False, downsampling=None):
        super(ConvBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcut = shortcut
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, input):

        residual = input
        output = self.conv1(input)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.shortcut:
            if self.downsampling is not None:
                residual = self.downsampling(input)
            output += residual

        output = self.relu2(output)

        return output


class VDCNN(nn.Module):
    def __init__(self, embedding_dim=768, base_num_features=256,
                 n_fc_neurons=256, depth=9, kernel_size=3,
                 padding=1, shortcut=True, dropout=0.2):
        super(VDCNN, self).__init__()

        layers = []
        fc_layers = []

        # num_conv_block
        if depth == 4:
            num_conv_block = [0, 0]
        elif depth == 12:
            num_conv_block = [2, 2]
        elif depth == 24:
            num_conv_block = [5, 5]

        # Project embedding_dim into base_num_features
        layers.append(nn.Conv1d(embedding_dim, base_num_features, kernel_size=1))

        # ConvBlock 1
        layers.append(ConvBlock(base_num_features, base_num_features, kernel_size, padding, shortcut=shortcut))
        for _ in range(num_conv_block[0]):
            layers.append(ConvBlock(base_num_features, base_num_features, kernel_size, padding, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=padding))
        layers.append(nn.Dropout(dropout))

        # ConvBlock 2
        ds = nn.Sequential(nn.Conv1d(base_num_features, 2 * base_num_features, 1, stride=1, bias=False),
                           nn.BatchNorm1d(2 * base_num_features))
        layers.append(ConvBlock(base_num_features, 2 * base_num_features, kernel_size, padding, shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[1]):
            layers.append(ConvBlock(2 * base_num_features, 2 * base_num_features, kernel_size, padding, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.Dropout(dropout))

        # Pooling and Linear
        layers.append(nn.AdaptiveMaxPool1d(8))
        layers.append(nn.Dropout(dropout))
        fc_layers.extend([nn.Linear(8 * 2 * base_num_features, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input, attention_mask):
        output = input.transpose(1, 2)
        output = self.layers(output)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)
        return output
