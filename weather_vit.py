import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.patch_conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        # return: (B, num_patches, embed_dim)
        return self.patch_conv(x).flatten(2).permute(0, 2, 1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, N, embed_dim)
        # return: (B, N, embed_dim)
        qkv = self.qkv(x)
        q, k ,v = torch.chunk(qkv, 3, dim=-1)
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads ,self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (torch.softmax(q@k.transpose(-2, -1)/math.sqrt(self.head_dim), dim=-1)) @ v # (B, num_heads, N, head_dim)
        return self.output_proj(attn.transpose(1, 2).flatten(2))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, x):
        # x: (B, N, embed_dim)
        # return: (B, N, embed_dim)
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, x):
        # x = x + Attention(LayerNorm(x))
        # x = x + FeedForward(LayerNorm(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class WeatherViT(nn.Module):
    def __init__(self, in_channels=12, out_channels=6, img_h=105, img_w=281,
                 patch_h=3, patch_w=3, embed_dim=128, num_heads=4,
                 num_layers=4, ff_hidden_dim=512):
        super().__init__()
        self.H_pad = math.ceil(img_h/patch_h) * patch_h
        self.W_pad = math.ceil(img_w/patch_w) * patch_w
        self.pad_h = self.H_pad - img_h
        self.pad_w = self.W_pad - img_w
        self.grid_h = (self.H_pad // patch_h)
        self.grid_w = (self.W_pad // patch_w)
        self.num_patches = self.grid_h * self.grid_w
        # Store these for unpatchify
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.out_channels = out_channels
        self.img_h = img_h
        self.img_w = img_w

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, (patch_h, patch_w))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for i in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.prediction_head = nn.Linear(embed_dim, out_channels*patch_h*patch_w)

    def unpatchify(self, x):
        """
        x: (B, num_patches, out_channels * patch_h * patch_w)
        return: (B, out_channels, H, W)
        """
        x = x.reshape(x.shape[0], self.grid_h, self.grid_w, self.out_channels, self.patch_h, self.patch_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(x.shape[0], self.out_channels, self.H_pad, self.W_pad)
        return x
    def forward(self, x):
        # x: (B, in_channels, H, W)
        # 1. Pad input if needed
        # 2. Patch embed → (B, num_patches, embed_dim)
        # 3. Add positional embedding
        # 4. Pass through transformer blocks
        # 5. Final layer norm
        # 6. Prediction head → (B, num_patches, out_channels * pH * pW)
        # 7. Unpatchify → (B, out_channels, H, W)
        # 8. Crop padding if added
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        h = self.patch_embedding(x)
        h = h + self.pos_embedding
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = self.prediction_head(h) # (B, num_patches, out_channels * patch_h * patch_w)
        h = self.unpatchify(h) #(B, out_channels, H_pad, W_pad)
        return h[:, :, 0:self.img_h, 0:self.img_w]