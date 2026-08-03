
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    HAS_WEIGHTS = True
except ImportError:
    from torchvision.models import efficientnet_b3
    HAS_WEIGHTS = False

from src.config import NUM_CLASSES, IMG_SIZE

logger = logging.getLogger("brain_pipeline")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SUB-MODULES (identik dengan notebook training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1536, patch_size=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_attn:
            return x, attn  # attn: [B, num_heads, N, N] -- dipakai buat heatmap
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=1536, vit_dim=768, fusion_dim=512, dropout=0.3):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, cnn_feat, vit_feat):
        c = self.cnn_proj(cnn_feat)
        v = self.vit_proj(vit_feat)
        w = self.attn(torch.cat([c, v], dim=-1))
        fused = w[:, 0:1] * c + w[:, 1:2] * v
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL UTAMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BrainHybridModel(nn.Module):
    """
    Nama class TETAP 'BrainHybridModel' (biar main.py/explainability.py
    tidak perlu diubah), tapi ISI-nya sekarang identik dengan
    HybridViTEfficientNet di notebook training.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 efficientnet_variant: str = "b3",
                 vit_embed_dim: int = 768,
                 vit_num_heads: int = 12,
                 vit_num_layers: int = 6,
                 fusion_dim: int = 512,
                 dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()

        # 1. CNN backbone (EfficientNet-B3)
        if HAS_WEIGHTS:
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        else:
            backbone = efficientnet_b3(pretrained=True)
        self.features = backbone.features
        self.cnn_out = 1536

        # 2. ViT branch (custom, dibangun dari feature map CNN)
        self.patch_embed = PatchEmbedding(self.cnn_out, patch_size=1,
                                           embed_dim=vit_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        num_patches = (IMG_SIZE // 32) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(vit_embed_dim, vit_num_heads, dropout=dropout)
            for _ in range(vit_num_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_embed_dim)

        # 3. Fusion
        self.fusion = CrossModalAttentionFusion(
            cnn_dim=self.cnn_out, vit_dim=vit_embed_dim,
            fusion_dim=fusion_dim, dropout=dropout)

        # 4. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Freeze/unfreeze backbone (opsional, hanya relevan kalau fine-tune ulang)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        cnn_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        patches = self.patch_embed(feat_map)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.vit_norm(tokens)
        vit_feat = tokens[:, 0]
        fused = self.fusion(cnn_feat, vit_feat)
        logits = self.classifier(fused)
        return logits

    def forward_with_attention(self, x: torch.Tensor):
        """
        Sama seperti forward(), tapi juga mengembalikan attention weights
        dari layer Transformer TERAKHIR -- dipakai untuk bikin heatmap
        'Peta Atensi Model' di explainability.py.

        Return:
            logits: [B, num_classes]
            last_attn: [B, num_heads, seq_len, seq_len]
        """
        feat_map = self.features(x)
        cnn_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        patches = self.patch_embed(feat_map)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                tokens, last_attn = blk(tokens, return_attn=True)
            else:
                tokens = blk(tokens)
        tokens = self.vit_norm(tokens)
        vit_feat = tokens[:, 0]
        fused = self.fusion(cnn_feat, vit_feat)
        logits = self.classifier(fused)
        return logits, last_attn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def print_model_info(model: nn.Module, device: str):
    try:
        from src.config import BATCH_SIZE, LR, EPOCHS, IMG_SIZE
    except ImportError:
        BATCH_SIZE, LR, EPOCHS, IMG_SIZE = 16, 5e-5, 30, 224

    total, trainable, frozen = count_parameters(model)
    sep70 = "=" * 70
    logger.info("\n" + sep70)
    logger.info("  INFORMASI MODEL")
    logger.info(sep70)
    col = 30
    fields = [
        ("Model Name",              "EfficientNet-B3 + Custom Vision Transformer (Hybrid)"),
        ("Architecture",            "EfficientNet-B3 Features + Custom ViT (6 layer) -> Cross-Modal Fusion -> MLP Head"),
        ("Jumlah Parameter",        f"{total:,}"),
        ("Trainable Parameter",     f"{trainable:,}"),
        ("Non-Trainable Parameter", f"{frozen:,}"),
        ("Image Size",              f"{IMG_SIZE} x {IMG_SIZE} px"),
        ("Batch Size",              str(BATCH_SIZE)),
        ("Learning Rate",           str(LR)),
        ("Epoch",                   str(EPOCHS)),
        ("Device",                  device.upper()),
    ]
    for k, v in fields:
        logger.info(f"  {k:<{col}}: {v}")
    logger.info("")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QUICK SANITY CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    model = BrainHybridModel()
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = model(dummy)
    print(f"Hybrid Model OK! Output shape: {out.shape}")
    total, trainable, frozen = count_parameters(model)
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")
