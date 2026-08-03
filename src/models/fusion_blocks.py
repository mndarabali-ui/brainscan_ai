import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Fuses features from two different sources (e.g. CNN features and ViT/Transformer features)
    using Cross-Attention mechanism.
    """
    def __init__(self, d_model=1280, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query, key, value shape: [batch_size, seq_len, d_model]
        attn_output, _ = self.multihead_attn(query, key, value)
        x = query + self.dropout(attn_output)
        x = self.norm(x)
        return x

class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation of features followed by linear projection.
    """
    def __init__(self, in_features1, in_features2, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features1 + in_features2, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=-1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
