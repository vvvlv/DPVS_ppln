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

from ..registry import register_model

# patch embedding -> cut the image into non-overlapping squares and for each square learn a small “summary vector” of length 'embed_dim'
# embed_dim vectors are tokens for transformer
class LinearEmbedding(nn.Module):
    def __init__(self, image_size=256, patch_size=4, in_chans=3, embedding_dim=48):
        # image shape: (batch size, channel, height, width)
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
        # self.proj(x) -> Shape: (batch size, embedding_dim, grid_size, grid_size)
        # .flatten(2) -> Flattens the last two dimensions (grid_size, grid_size) into one
        # -> (batch size, embedding_dim, grid_size * grid_size)
        # .transpose(1, 2) -> Shape: (batch size, num_patches, embedding_dim) -> swaps channel and sequence dimensions
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
        # num_windows_total = num_windows * B (batch size)
        # window_num_tokens = window_size_h * window_size_w (number of tokens in each window) -> window size * window size
        num_windows_total, window_num_tokens, channel = x.shape

        # Shape: (num_windows_total, window_num_tokens, 3*channel)
        query_key_value = self.query_key_value(x)
        dim_size_per_head = channel // self.att_heads_num_layer
        query_key_value = query_key_value.reshape(
            num_windows_total, window_num_tokens, 
            3, 
            self.att_heads_num_layer, 
            dim_size_per_head
        )
        # Shape: (3, num_windows_total, att_heads_num_layer, window_num_tokens, dim_size_per_head) -> 3 because qkv
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)

        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # compute attention logits per head
        # query * dim_size_per_head^(-0.5)
        query = query * self.scale
        # query^Tkey
        # Shape: (num_windows_total, att_heads_num_layer, window_num_tokens, window_num_tokens) 
        # -> window_num_tokensxwindow_num_tokens matrix of attention scores per batch
        attention = (query @ key.transpose(-2, -1))

        # apply attention mask (for shifted windows)
        if mask is not None:
            # mask: (num_windows_per_image, window_num_tokens, window_num_tokens)
            num_windows_per_image = mask.shape[0]
            # separate batch and windows
            batch_size = num_windows_total // num_windows_per_image
            attention = attention.view(
                    batch_size, 
                    num_windows_per_image, 
                    self.att_heads_num_layer, 
                    window_num_tokens, 
                    window_num_tokens
                )
            # add the mask with broadcasting
            attention = attention + mask.unsqueeze(1).unsqueeze(0)  
            # flatten back to original shape
            # from: (num_windows_total_one_batch, num_windows_per_image, att_heads_num_layer, window_num_tokens, window_num_tokens)
            # to: (num_windows_total, att_heads_num_layer, window_num_tokens, window_num_tokens)
            attention = attention.view(-1, self.att_heads_num_layer, window_num_tokens, window_num_tokens)

        attention = attention.softmax(dim=-1)

        # Shape: (num_windows_total, att_heads_num_layer, window_num_tokens, dim_size_per_head)
        x = attention @ value
        # Shape: (num_windows_total, window_num_tokens, att_heads_num_layer, dim_size_per_head)
        x = x.transpose(1, 2)
        # Shape: (num_windows_total, window_num_tokens, channel) -> channel = att_heads_num_layer * dim_size_per_head
        x = x.reshape(num_windows_total, window_num_tokens, channel)
        x = self.projection_layer(x)
        return x
    
def window_partition(image_masks, window_size):
    batch_size, heights, widths, channels = image_masks.shape

    num_windows_h = heights // window_size
    num_windows_w = widths // window_size

    image_masks = image_masks.view(
        batch_size,
        num_windows_h, 
        # window size height
        window_size,
        num_windows_w, 
        # window size width
        window_size,
        channels,
    )

    # from shape: (batch_size, num_windows_h, window_size_h, num_windows_w, window_size_w, channels) 
    # to shape:   (batch_size, num_windows_h, num_windows_w, window_size_h, window_size_w, channels) 
    windows = image_masks.permute(0, 1, 3, 2, 4, 5).contiguous()
    # -1 means: num_windows_total = batch_size * num_windows_h * num_windows_w
    # Shape: (num_windows_total, window_size, window_size, channels)
    windows = windows.view(-1, window_size, window_size, channels)
    return windows

def window_reverse(windows, window_size, height, width):
    # (num_windows * batch_size)
    total_num_windows = windows.shape[0] 
    # windows_per_image = (height // window_size) * (width // window_size)
    # batch_size = total_num_windows // windows_per_image
    windows_per_image = (height * width / window_size / window_size)
    batch_size = int(total_num_windows / windows_per_image)
    # Shape: (batch size, num_windows_h, num_windows_w, window_size, window_size, channel (-1 taken from original window))
    x = windows.view(
        batch_size,
        height // window_size, width // window_size,
        window_size,
        window_size, 
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch_size, height, width, -1)
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
        assert 0 <= self.shift_size < self.window_size, \
            f"Invalid shift size: {self.shift_size}"

        # swin transformer block components
        self.norm_1 = nn.LayerNorm(normalized_shape=self.dimensions)
        # # multi-head self-attention
        self.attention = WindowAttention(
            dimensions=self.dimensions, 
            window_size=self.window_size, 
            att_heads_num_layer=self.att_heads_num_layer
        )
        self.norm_2 = nn.LayerNorm(normalized_shape=self.dimensions)

        mlp_hidden_dimension = int(self.dimensions * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dimensions, mlp_hidden_dimension),
            nn.GELU(),
            nn.Linear(mlp_hidden_dimension, dimensions),
        )

        if self.shift_size > 0:
            # create a fake image
            image_masks = torch.zeros((1, self.patch_resolution[0], self.patch_resolution[1], 1))

            # 3x3 grid
            h_slices = (
                # top 
                slice(0, -self.window_size),
                # middle
                slice(-self.window_size, -self.shift_size),
                # bottom
                slice(-self.shift_size, None),
            )
            w_slices = (
                # left
                slice(0, -self.window_size),
                # middle
                slice(-self.window_size, -self.shift_size),
                # right
                slice(-self.shift_size, None),
            )

            i = 0
            for height in h_slices:
                for width in w_slices:
                    # h0, w0 -> top,left
                    # h0, w1 -> top, middle
                    # h0, w2 -> top, right
                    # ...
                    # Shape: (B, H, W, C)
                    image_masks[:, height, width, :] = i
                    i += 1

            mask_windows = window_partition(image_masks, self.window_size)
            # Shape: (number of windows, token resolution)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # mask_windows.unsqueeze(1) -> (nW, 1, N) <- adds new dim in pos 1 (row)
            # mask_windows.unsqueeze(2) -> (nW, N, 1)  <- adds new dim in pos (col)
            # same-region tokens -> difference = 0
            # Different-region tokens -> difference ≠ 0
            att_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            # forbidden positions become -100, to later get almost zero probability
            # effect: attention is blocked for those pairs of tokens
            att_mask = att_mask.masked_fill(att_mask != 0, float(-100.0))
            # allowed positions, so no effect in probs
            att_mask = att_mask.masked_fill(att_mask == 0, float(0.0))
        else:
            att_mask = None

        self.register_buffer("att_mask", att_mask)

    def forward(self, x):
        batch_size, num_tokens, channel = x.shape
        height, width = self.patch_resolution
        assert num_tokens == height * width, \
            f"Input length {num_tokens} does not match H*W = {height}*{width}"

        # attention branch
        # save the original input for the residual connection later
        shortcut = x
        x = self.norm_1(x)
        # from shape: (batch_size, height*width, channel) to (batch_size, height, width, C)
        x = x.view(batch_size, height, width, channel)

        # cyclic shift (for SW-MSA)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                # move the feature map 'up' by shift_size pixels and 'left' by shift_size pixels
                shifts=(-self.shift_size, -self.shift_size),
                # shift along height and width
                dims=(1, 2),
            )
        # W-MSA
        else:
            shifted_x = x

        # partition windows: 
        # cuts each (height, width) feature map into non-overlapping windows of window size
        # from (batch size, height, width, channel) to (num_windows*batch size, window_size, window_size, channel)
        x_windows = window_partition(shifted_x, self.window_size) 
        # flatten
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channel)

        # so each row in this batch is one window, seen as number of tokens

        att_windows = self.attention(x_windows, mask=self.att_mask)

        # merge windows
        att_windows = att_windows.view(-1, self.window_size, self.window_size, channel)
        shifted_x = window_reverse(att_windows, self.window_size, height, width) 

        # reverse shift (shift back to bring tokens back together)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(batch_size, num_tokens, channel)
        x = shortcut + x   

        # mlp
        shortcut = x
        x = self.norm_2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x
    
class PatchMerging(nn.Module):
    def __init__(self, patch_resolution, dimension):
        super().__init__()
        self.patch_resolution = patch_resolution
        self.dimension = dimension
        # compresses the vector from 4-dimension to the 2-dimension (the channels)
        self.linear = nn.Linear(4 * self.dimension, 2 * self.dimension, bias=False)
        # stabilize
        self.norm = nn.LayerNorm(4 * self.dimension)

    def forward(self, x):
        batch_size, patch_resolution, channel = x.shape
        height, width = self.patch_resolution
        assert patch_resolution == height * width, \
                f"Input length {patch_resolution} does not match H*W = {height}*{width}"

        x = x.view(batch_size, height, width, channel)

        # take 2x2 patches and concatenate their channels
        # 0::2 means indices 0, 2, 4, ...
        # each of these has shape: batch size, height/2, width/2, channel
        # -> because of skipping every second row and column
        # x_0: pixels at (even row, even col)
        x_0 = x[:, 0::2, 0::2, :]
        # x_1: pixels at (odd row, even col)
        x_1 = x[:, 1::2, 0::2, :]
        # x_2: pixels at (even row, odd col)
        x_2 = x[:, 0::2, 1::2, :]
        # x_3: pixels at (odd row, odd col)
        x_3 = x[:, 1::2, 1::2, :]

        # visually:
        # x_0: (0,0) x_2:(0,1)
        # x_1: (1,0) x_3:(1,1)

        # merge 4 tokens into one
        x = torch.cat([x_0, x_1, x_2, x_3], dim=-1)
        x = x.view(batch_size, -1, 4 * channel)

        x = self.norm(x)
        x = self.linear(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, patch_resolution, dimension):
        super().__init__()
        self.patch_resolution = patch_resolution
        self.dimension = dimension

        # expand channels -> 4*channels (so we can rearrange 2x2)
        self.linear = nn.Linear(self.dimension , 4 * self.dimension , bias=False) 
        self.norm = nn.LayerNorm(self.dimension)

    def forward(self, x):
        batch_size, num_tokens, channel = x.shape
        height, width = self.patch_resolution
        assert num_tokens == height * width, \
                f"Input length {num_tokens} does not match H*W = {height}*{width}"
        
        x = self.linear(x)

        # put token onto grid
        x = x.view(batch_size, height, width, 4 * channel)
        # from vextor of 2x3 groups of channels to 2x2 grid
        x = x.view(batch_size, height, width, 2, 2, channel) 
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() 
        x = x.view(batch_size, 2 * height, 2 * width, channel)  

        # back to token
        x = x.view(batch_size, 4 * height * width, channel) 

        x = self.norm(x)

        return x, 2 * height, 2 * width


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

        act_name = config.get("final_activation", None)
        if act_name == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif act_name == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # divide image into patches and embed them
        self.patch_embed = LinearEmbedding(
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
            height_layer = heights[layer_idx]
            width_layer = widths[layer_idx]
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
                        patch_resolution=(height_layer, width_layer),
                        att_heads_num_layer=att_heads_num_layer,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                    )
                )
            self.encoder_layers.append(layer_blocks)

            # (downsample) except of bottleneck
            if layer_idx < self.num_layers - 1:
                self.patch_merging_layers.append(
                    PatchMerging(patch_resolution=(height_layer, width_layer), dimension=dimension_layer)
                )

        self.upsampling_layers = nn.ModuleList()
        self.concat_linear = nn.ModuleList()  
        self.decoder_layers = nn.ModuleList()

        # -1 because of skip connection
        for layer_idx in range(self.num_layers - 1):
            # index from deepest to higher (except of bottleneck)
            depth_idx = (self.num_layers - 2) - layer_idx

            # +1 -> what we’re upsampling from
            self.upsampling_layers.append(
                PatchExpand(
                    patch_resolution=(heights[depth_idx + 1], widths[depth_idx + 1]),
                    dimension=dimensions[depth_idx + 1],
                )
            )

            # skip connection
            self.concat_linear.append(
                nn.Linear(
                    in_features=dimensions[depth_idx + 1] + dimensions[depth_idx],
                    out_features=dimensions[depth_idx],
                )
            )

            decoder_blocks = nn.ModuleList()
            for block_idx in range(depths[depth_idx]):
                shift_size = 0 if (block_idx % 2 == 0) else window_size // 2
                decoder_blocks.append(
                    TinySwinTransformerBlock(
                        dimensions=dimensions[depth_idx],
                        patch_resolution=(heights[depth_idx], widths[depth_idx]),
                        att_heads_num_layer=att_head_num[depth_idx],
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                    )
                )
            self.decoder_layers.append(decoder_blocks)

        # tokens to segmentation layer
        self.final_layer = nn.Conv2d(
            in_channels=dimensions[0],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        batch_size, input_channels, height, width = x.shape
        assert height == self.image_size and width == self.image_size, \
            f"Input image size ({height}x{width}) must match config image_size {self.image_size}."

        x = self.patch_embed(x)

        skip_connections = []
        for layer_idx in range(self.num_layers):
            # encoder - swin blocks
            for block in self.encoder_layers[layer_idx]:
                x = block(x)

            if layer_idx < self.num_layers - 1:
                skip_connections.append(x)
                # downsample
                x = self.patch_merging_layers[layer_idx](x)

        # bottleneck
        height_tokens = self.patch_embed.grid_size // (2 ** (self.num_layers - 1))
        width_tokens = height_tokens

        # upsample from layer + 1 to layer (deepper vs higher)
        for layer_idx in range(self.num_layers - 1):
            x, height_tokens, width_tokens = self.upsampling_layers[layer_idx](x)

            skip_layer = self.num_layers - 2 - layer_idx
            skip = skip_connections[skip_layer] 

            # concat channels
            x = torch.cat([x, skip], dim=-1) 
            x = self.concat_linear[layer_idx](x) 

            # decoder - swin blocks
            for block in self.decoder_layers[layer_idx]:
                x = block(x)

        # final layer
        final_height = self.patch_embed.grid_size
        final_width = final_height
        batch_size, patch_resolution, channel = x.shape
        assert patch_resolution == final_height * final_width, \
                f"Final token length {patch_resolution} != {final_height}*{final_width}"

        # extract tokens to feature map
        # to (batch_size, channel, final_height, final_width)
        x = x.view(batch_size, final_height, final_width, channel).permute(0, 3, 1, 2) 

        # upsample feature map back to original image resolution
        x = torch.nn.functional.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        x = self.final_layer(x) 

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


            


        



