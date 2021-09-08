from models.base import *

class SelfAttention(nn.Module):
    def __init__(self, feature_size, dropout_rate=0.2):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size
        self.dropout_rate = dropout_rate
        self.attention_scorer = FFNNModule(
                                    input_size=feature_size,
                                    hidden_sizes=[feature_size // 2],
                                    output_size=1,
                                    dropout=dropout_rate
                                )

    def forward(self, seq, attention_mask):
        attns = self.attention_scorer(torch.transpose(seq, 1, 2))
        attn_probs = torch.softmax(attns, dim=-1)
        attn_probs = attn_probs * attention_mask
        attn_probs = attn_probs / torch.sum(attn_probs, dim=-1, keepdim=True)

        output = torch.sum(seq * attn_probs.unsqueeze(1), dim=-1)
        return output
