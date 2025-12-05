"""
The implementation of Swin Unet is inspired by:

Article:
  Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang,
  Qi Tian, and Manning Wang,
  "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation".

Original Swin-Unet implementation:
  https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
"""

import torch.nn as nn
import torch

#### The code from Original Swin-Unet implementation
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
####

#### The code from Original Swin-Unet implementation
def window_reverse(windows, window_size, height, width):
    total_num_windows = windows.shape[0]
    num_windows_h = height // window_size
    num_windows_w = width // window_size
    windows_per_image = num_windows_h * num_windows_w
    batch_size = total_num_windows // windows_per_image

    x = windows.view(
        batch_size,
        num_windows_h, num_windows_w,
        window_size, window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x
###

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
    

#### Modified code from Original Swin-Unet implementation
class TinySwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dimensions,
        patch_resolution,
        att_heads_num_layer,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        dropout_rate: float = 0.0,
    ):
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
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_mlp = nn.Dropout(dropout_rate)

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
        x = shortcut + self.dropout_attn(x)   

        # mlp
        shortcut = x
        x = self.norm_2(x)
        x = self.mlp(x)
        x = shortcut + self.dropout_mlp(x)

        return x
####

#### The code from Original Swin-Unet implementation
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
####

#### The code from Original Swin-Unet implementation
class PatchExpand(nn.Module):
    def __init__(self, patch_resolution, dimension):
        super().__init__()
        self.patch_resolution = patch_resolution
        self.dimension = dimension

        # expand channels -> 4*channels (so we can rearrange 2x2)
        self.linear = nn.Linear(self.dimension , 4 * self.dimension , bias=False) 
        self.norm = nn.LayerNorm(self.dimension)
#### 

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