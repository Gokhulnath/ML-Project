"""
Swin-UNet: UNet-like Pure Transformer for Medical Image Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, (self.window_size, self.window_size))
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA
        attn_windows = self.attn(x_windows, mask=None)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""
    
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer"""
    
    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)
        self.dim_scale = dim_scale
    
    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        # Manual expansion instead of einops rearrange
        # Expand from (B, H, W, C) to (B, H*dim_scale, W*dim_scale, C/(dim_scale^2))
        C_new = C // (self.dim_scale ** 2)
        x_expanded = x.view(B, H, W, self.dim_scale, self.dim_scale, C_new)
        x_expanded = x_expanded.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x_expanded.view(B, H * self.dim_scale, W * self.dim_scale, C_new)
        x = x.view(B, -1, C_new)
        x = self.norm(x)
        
        return x


class Mlp(nn.Module):
    """MLP"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class BasicLayer(nn.Module):
    """Basic Swin Transformer layer"""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Pure Transformer architecture for medical image segmentation
    """
    
    def __init__(self, img_size=224, in_channels=3, n_classes=9, embed_dim=96, 
                 depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super(SwinUNet, self).__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, in_channels=in_channels, embed_dim=embed_dim)
        patches_resolution = img_size // 4  # patch_size is 4
        self.patches_resolution = (patches_resolution, patches_resolution)
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution // (2 ** i_layer),
                    patches_resolution // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.
            )
            self.encoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                patch_merge = PatchMerging(
                    input_resolution=(
                        patches_resolution // (2 ** i_layer),
                        patches_resolution // (2 ** i_layer)
                    ),
                    dim=int(embed_dim * 2 ** i_layer)
                )
                self.encoder_layers.append(patch_merge)
        
        # Bottleneck - track the resolution after encoder
        bottleneck_resolution = patches_resolution // (2 ** self.num_layers)
        self.bottleneck = BasicLayer(
            dim=int(embed_dim * 2 ** self.num_layers),
            input_resolution=(bottleneck_resolution, bottleneck_resolution),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=min(window_size, bottleneck_resolution),  # Ensure window_size <= resolution
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - i_layer)),
                    input_resolution=(
                        patches_resolution // (2 ** (self.num_layers - i_layer)),
                        patches_resolution // (2 ** (self.num_layers - i_layer))
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.
                )
            else:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - i_layer)),
                    input_resolution=(
                        patches_resolution // (2 ** (self.num_layers - i_layer)),
                        patches_resolution // (2 ** (self.num_layers - i_layer))
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.
                )
            self.decoder_layers.append(layer)
            if i_layer < self.num_layers - 1:
                patch_expand = PatchExpanding(
                    input_resolution=(
                        patches_resolution // (2 ** (self.num_layers - i_layer)),
                        patches_resolution // (2 ** (self.num_layers - i_layer))
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - i_layer))
                )
                self.decoder_layers.append(patch_expand)
        
        # Final projection
        self.final_proj = nn.Conv2d(embed_dim, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, H*W, embed_dim
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        
        # Encoder
        encoder_features = []
        x_reshaped = x.view(B, H * W, C)
        current_res = H  # Track current resolution
        
        for layer in self.encoder_layers:
            if isinstance(layer, BasicLayer):
                # Update input resolution for BasicLayer
                layer.input_resolution = (current_res, current_res)
                layer.window_size = min(layer.window_size, current_res)
                x_reshaped = layer(x_reshaped)
                H_layer = W_layer = int(x_reshaped.shape[1] ** 0.5)
                encoder_features.append(x_reshaped.view(B, H_layer, W_layer, -1))
                current_res = H_layer
            else:  # PatchMerging
                x_reshaped = layer(x_reshaped)
                current_res = current_res // 2
        
        # Bottleneck
        bottleneck_res = current_res
        self.bottleneck.input_resolution = (bottleneck_res, bottleneck_res)
        self.bottleneck.window_size = min(self.bottleneck.window_size, bottleneck_res)
        x_reshaped = self.bottleneck(x_reshaped)
        
        # Decoder with skip connections
        decoder_idx = 0
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, PatchExpanding):
                x_reshaped = layer(x_reshaped)
                # Update spatial dimensions after expansion
                H_dec = W_dec = int(x_reshaped.shape[1] ** 0.5)
            else:  # BasicLayer
                # Add skip connection from corresponding encoder layer
                if decoder_idx < len(encoder_features):
                    skip_idx = len(encoder_features) - 1 - decoder_idx
                    if skip_idx >= 0 and skip_idx < len(encoder_features):
                        skip_feat = encoder_features[skip_idx]
                        skip_flat = skip_feat.view(B, -1, skip_feat.shape[-1])
                        # Get current spatial size
                        H_curr = W_curr = int(x_reshaped.shape[1] ** 0.5)
                        H_skip = W_skip = int(skip_flat.shape[1] ** 0.5)
                        # Ensure dimensions match - resize if needed
                        if x_reshaped.shape[-1] == skip_flat.shape[-1]:
                            if H_curr == H_skip:
                                x_reshaped = x_reshaped + skip_flat
                            else:
                                # Resize skip connection to match
                                skip_2d = skip_flat.view(B, H_skip, W_skip, -1).permute(0, 3, 1, 2)
                                skip_2d = F.interpolate(skip_2d, size=(H_curr, W_curr), mode='bilinear', align_corners=True)
                                skip_resized = skip_2d.permute(0, 2, 3, 1).view(B, H_curr * W_curr, -1)
                                x_reshaped = x_reshaped + skip_resized
                x_reshaped = layer(x_reshaped)
                decoder_idx += 1
        
        # Final projection
        H_final = W_final = int(x_reshaped.shape[1] ** 0.5)
        x_final = x_reshaped.view(B, H_final, W_final, -1).permute(0, 3, 1, 2)
        x_final = F.interpolate(x_final, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        output = self.final_proj(x_final)
        
        return output


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
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

