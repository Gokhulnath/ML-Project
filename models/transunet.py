"""
TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_resnet import resnet50_features
import math


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, 
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DecoderCup(nn.Module):
    """Decoder with skip connections"""
    
    def __init__(self, encoder_channels, decoder_channels, n_classes):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.encoder_channels = encoder_channels
        
        # Decoder blocks
        self.decoder1 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(decoder_channels[0] + encoder_channels[-2], decoder_channels[1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(decoder_channels[1] + encoder_channels[-3], decoder_channels[2], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + encoder_channels[-4], decoder_channels[3], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(decoder_channels[3], n_classes, kernel_size=1)
    
    def forward(self, features):
        # features: list of encoder features [x1, x2, x3, x4, x5]
        x = features[-1]  # Bottleneck
        x = self.decoder1(x)
        x = self.upsample(x)
        x = torch.cat([x, features[-2]], dim=1)
        x = self.decoder2(x)
        x = self.upsample(x)
        x = torch.cat([x, features[-3]], dim=1)
        x = self.decoder3(x)
        x = self.upsample(x)
        x = torch.cat([x, features[-4]], dim=1)
        x = self.decoder4(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid CNN-Transformer architecture
    Uses ResNet-50 encoder + Vision Transformer + CNN decoder
    """
    
    def __init__(self, img_size=224, in_channels=3, n_classes=9, 
                 embed_dim=768, depth=12, num_heads=12):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        
        # ResNet-50 encoder
        resnet = resnet50_features(pretrained=False)  # Use simple ResNet to avoid import issues
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4
        
        # Encoder channel dimensions: [64, 256, 512, 1024, 2048]
        encoder_channels = [64, 256, 512, 1024, 2048]
        
        # Vision Transformer at bottleneck
        self.transformer = VisionTransformer(
            img_size=img_size // 16,  # After 4 downsampling stages
            patch_size=1,
            in_channels=encoder_channels[-1],
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Reshape transformer output back to feature map
        self.transformer_proj = nn.Conv2d(embed_dim, encoder_channels[-1], kernel_size=1)
        
        # Projection layer for transformer input
        self.bottleneck_proj = nn.Linear(encoder_channels[-1], embed_dim)
        
        # Decoder
        decoder_channels = [512, 256, 128, 64]
        self.decoder = DecoderCup(encoder_channels, decoder_channels, n_classes)
    
    def forward(self, x):
        # CNN Encoder (ResNet-50)
        x1 = self.resnet_conv1(x)
        x1 = self.resnet_bn1(x1)
        x1 = self.resnet_relu(x1)
        x1 = self.resnet_maxpool(x1)
        
        x2 = self.resnet_layer1(x1)
        x3 = self.resnet_layer2(x2)
        x4 = self.resnet_layer3(x3)
        x5 = self.resnet_layer4(x4)
        
        encoder_features = [x1, x2, x3, x4, x5]
        
        # Transformer at bottleneck
        bottleneck = encoder_features[-1]  # B, 2048, H/16, W/16
        B, C, H, W = bottleneck.shape
        
        # Project to transformer embed_dim
        bottleneck_flat = bottleneck.flatten(2).transpose(1, 2)  # B, H*W, C
        bottleneck_proj = self.bottleneck_proj(bottleneck_flat)
        
        # Add positional encoding (interpolate if needed)
        pos_embed = self.transformer.pos_embed[:, 1:, :]  # Remove cls token
        if pos_embed.shape[1] != bottleneck_proj.shape[1]:
            # Interpolate positional encoding to match spatial dimensions
            pos_h = pos_w = int(pos_embed.shape[1] ** 0.5)
            pos_embed_2d = pos_embed.transpose(1, 2).reshape(1, -1, pos_h, pos_w)
            pos_embed_2d = F.interpolate(pos_embed_2d, size=(H, W), mode='bilinear', align_corners=True)
            pos_embed = pos_embed_2d.flatten(2).transpose(1, 2)
        
        bottleneck_proj = bottleneck_proj + pos_embed
        bottleneck_proj = self.transformer.pos_drop(bottleneck_proj)
        
        # Process through transformer blocks
        for block in self.transformer.blocks:
            bottleneck_proj = block(bottleneck_proj)
        bottleneck_proj = self.transformer.norm(bottleneck_proj)
        
        # Reshape back: (B, H*W, embed_dim) -> (B, embed_dim, H, W)
        bottleneck_transformed = bottleneck_proj.transpose(1, 2).reshape(B, self.transformer.embed_dim, H, W)
        bottleneck_transformed = self.transformer_proj(bottleneck_transformed)
        
        # Decoder with skip connections
        encoder_features[-1] = bottleneck_transformed
        output = self.decoder(encoder_features)
        
        return output

