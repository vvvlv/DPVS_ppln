"""
The implementation of Swin Unet is inspired by:

Article:
  Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang,
  Qi Tian, and Manning Wang,
  "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation".

Original Swin-Unet implementation:
  https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
"""

# TODO: add normalization layers

import torch.nn as nn
import torch

# patch embedding -> cut the image into non-overlapping squares and for each square learn a small “summary vector” of length 'embed_dim'
# embed_dim vectors are tokens for transformer
class MakePatches(nn.Module):
    def __init__(self, image_size=256, patch_size=4, in_chans=3, embedding_dim=48):
        # image shape: (B, C, H, W)
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embedding_dim = embedding_dim
        # example: image - 256x256, patch_size - 4
        # grid_size = 256/4 = 64 
        self.grid_size = image_size // patch_size
        # num_patches = 64 * 64 = 4096
        self.num_patches = self.grid_size * self.grid_size

        # feature vector
        self.projection_layer = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # TODO: check this shape
        # self.proj(x) -> Shape: (B, embedding_dim, grid_size, grid_size)
        # .flatten(2) -> Flattens the last two dimensions (grid_size, grid_size) into one (B, embedding_dim, grid_size * grid_size)
        # .transpose(1, 2) -> Shape: (B, num_patches, embedding_dim) -> swaps channel and sequence dimensions
        x = self.projection_layer(x).flatten(2).transpose(1, 2)
        return x
    
class WindowAttention(nn.Module):
    def __init__(self, dimensions, window_size, att_heads_num_layer):
        super().__init__()
        self.dimensions = dimensions
        self.window_size = window_size
        self.att_heads_num_layer = att_heads_num_layer
        self.dim_size_per_head = dimensions // att_heads_num_layer
        self.scale = self.dim_size_per_head ** -0.5

        # concatenate: query, key, value -> self.dimensions*3
        self.query_key_value = nn.Linear(in_features=self.dimensions, out_features=self.dimensions * 3, bias=True)
        # feature vector
        self.projection_layer = nn.Linear(in_features=self.dimensions, out_features=self.dimensions)

    def forward(self, x, mask=None):
        # B_ = num_windows * B (batch size)
        # N = Wh * Ww (number of tokens in each window) -> window size * window size
        B_, N, C = x.shape

        # Shape: (B_, N, 3*C)
        query_key_value = self.query_key_value(x)
        # Shape: (B_, N, 3, att_heads_num_layer, dim_size_per_head)
        query_key_value = query_key_value.reshape(B_, N, 3, self.att_heads_num_layer, C // self.att_heads_num_layer)
        # Shape: (3, B_, att_heads_num_layer, N, dim_size_per_head) -> 3 because qkv
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)

        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # compute attention logits per head
        # query * dim_size_per_head^(-0.5)
        query = query * self.scale
        # query^Tkey
        # Shape: (B_, att_heads_num_layer, N, N) -> NxN matric of attention scores per batch
        attention = (query @ key.transpose(-2, -1))

        # apply attention mask (for shifted windows)
        if mask is not None:
            # mask: (num_windows_per_image, N, N)
            num_windows_per_image = mask.shape[0]
            # separate batch and windows
            attention = attention.view(B_ // num_windows_per_image, num_windows_per_image, self.att_heads_num_layer, N, N)
            # add the mask with broadcasting
            attention = attention + mask.unsqueeze(1).unsqueeze(0)  
            # flatten back to original shape
            # from: (B_one_batch, num_windows_per_image, att_heads_num_layer, N, N)
            # to: (B_, att_heads_num_layer, N, N)
            attention = attention.view(-1, self.att_heads_num_layer, N, N)

        attention = attention.softmax(dim=-1)

        # Shape: (B_, att_heads_num_layer, N, dim_size_per_head)
        x = attention @ value
        # Shape: (B_, N, att_heads_num_layer, dim_size_per_head)
        x = x.transpose(1, 2)
        # Shape: (B_, N, C) -> C = att_heads_num_layer * dim_size_per_head
        x = x.reshape(B_, N, C)
        x = self.projection_layer(x)
        return x


class TinySwinTransformerBlock(nn.Module):
    def __init__(self, dimensions, patch_resolution, att_heads_num_layer,
                 window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dimensions = dimensions
        self.patch_resolution = patch_resolution
        self.att_heads_num_layer = att_heads_num_layer
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Case when the feature map is smaller than the chosen window_size. It's not possible to split it into windows.
        if min(self.patch_resolution) <= self.window_size:
            # turn off shifting
            self.shift_size = 0
            # set window size to the minimum resolution -> one big window
            self.window_size = min(self.patch_resolution)

        # check for valid shift size
        assert 0 <= self.shift_size < self.window_size

        # swin transformer block components
        self.norm1 = nn.LayerNorm(normalized_shape=dimensions)
        # # multi-head self-attention
        self.attention = WindowAttention(
            dimensions=dimensions, 
            window_size=window_size, 
            att_heads_num_layer=att_heads_num_layer
        )
        self.norm2 = nn.LayerNorm(normalized_shape=dimensions)

        mlp_hidden_dim = int(dimensions * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dimensions, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dimensions),
        )

        if self.shift_size > 0:
            # create a fake image
            image_mask = torch.zeros((1, self.patch_resolution[0], self.patch_resolution[1], 1))
            
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)




@register_model('TinySwinUNet')
class TinySwinUNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        in_channels = config['in_channels']
        out_channels = config['out_channels']
        image_size = config["image_size"][0]
        patch_size = config.get("patch_size", 4)
        embedding_dim = config.get("embedding_dim", 48)
        depths = config.get("depths", [2, 2, 2, 2])
        att_head_num = config.get("att_head_num", [3, 6, 12, 24])
        window_size = config.get("window_size", 7)
        # multi layer perceptron ratio
        mlp_ratio = config.get("mlp_ratio", 4.0)

        self.img_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # divide image into patches and embed them
        self.patch_embed = MakePatches(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embedding_dim=embedding_dim,
        )

        self.num_layers = len(depths)

        # resolution stages
        heights = [self.patch_embed.grid_size // (2 ** idx) for idx in range(self.num_layers)]
        widths = [self.patch_embed.grid_size // (2 ** idx) for idx in range(self.num_layers)]

        # channel dimensions at different stages
        dimensions = [embedding_dim * (2 ** idx) for idx in range(self.num_layers)]

        # ensure the image can be downsampled properly
        total_downsample_factor = 2 ** (self.num_layers - 1)
        assert self.patch_embed.grid_size % total_downsample_factor == 0, \
            f"grid_size {self.patch_embed.grid_size} must be a multiple of {total_downsample_factor} for {self.num_layers} stages"
        
        # encoder (swin tranformer block) + patch merging layers (downsampling)
        self.encoder_layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            H_layer = heights[layer_idx]
            W_layer = widths[layer_idx]
            dimension_layer = dimensions[layer_idx]
            depth_layer = depths[layer_idx]
            att_heads_num_layer = att_head_num[layer_idx]

            layer_blocks = nn.ModuleList()
            # num of transformer blocks according to depth per encoder layer
            for block_idx in range(depth_layer):
                # to alternate between SW-MSA and W-MSA
                shift_size = 0 if (block_idx % 2 == 0) else window_size // 2
                layer_blocks.append(
                    TinySwinTransformerBlock(
                        dimensions=dimension_layer,
                        patch_resolution=(H_layer, W_layer),
                        att_heads_num_layer=att_heads_num_layer,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                    )
                )

            


        



