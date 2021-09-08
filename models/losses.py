# Code adapted from https://github.com/msight-tech/research-ms-loss
import torch
from torch import nn

class MultiSimilarityLoss(nn.Module):
    def __init__(self, configs):
        super(MultiSimilarityLoss, self).__init__()
        self.margin = 0.1
        self.thresh = configs['loss_thresh']

        self.scale_pos = configs['loss_scale_pos']
        self.scale_neg = configs['loss_scale_neg']

    def forward(self, sim_mat, labels):
        batch_size = sim_mat.size(0)

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss
