import torch
import torch.nn as nn
# to save memory by re-computing activations during backward.
import torch.utils.checkpoint as checkpoint
from einops import rearrange
# DropPath: stochastic depth (randomly drop entire residual branches).
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..registry import register_model

# Mixture-of-Experts MLP
class MoEFFNGating(nn.Module):
    # dim: feature dimension of the input (e.g. 96, 192, …).
    # hidden_dim: the inner dimension of each MLP (like the usual FFN expansion).
    # num_experts: how many separate MLPs (“experts”) we have.
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        # This is a gating network: it takes an input of shape (..., dim) and outputs another vector of shape (..., dim).
        # Later they will apply softmax on this, to turn it into weights.
        self.gating_network = nn.Linear(dim, dim)
        # creates a list of num_experts independent MLPs
        # So each expert takes a vector of size dim, passes it through:
        self.experts = nn.ModuleList([nn.Sequential(
            # Linear(dim → hidden_dim)
            nn.Linear(dim, hidden_dim),
            # GELU activation
            nn.GELU(),
            # Linear(hidden_dim → dim)
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])
            # The output of each expert has the same dimension as the input (dim), so they can be combined.

    def forward(self, x):
        # → applies a linear transformation to each vector in x.
        # result: (B, ..., dim)
        weights = self.gating_network(x)
        # mixture weights
        # normalizes across the last dimension (the dim dimension), so for each token weights[i,j,:] sums to 1
        # So for each position, weights is a probability distribution over the dim components.
        weights = torch.nn.functional.softmax(weights, dim=-1)
        # list of expert outputs
        outputs = [expert(x) for expert in self.experts]
        # (E, B, ..., dim)
        # where E is num_experts
        outputs = torch.stack(outputs, dim=0)
        # weighted sum over experts
        # weights: (B, ..., dim)
        # weights.unsqueeze(0): (1, B, ..., dim)
        # outputs: (E, B, ..., dim)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs
        # Broadcasting:
        # When you do weights.unsqueeze(0) * outputs, PyTorch broadcasts the (1, B, ..., dim) across the E dimension 
        # → each expert output is multiplied by the same weights for that token.
        # Then:
        # .sum(0) sums over E (expert dimension), giving: shape (B, ..., dim) again.
        # So mathematically:
        # output_b,t,d=∑d_e=1^E ​weights_b,t,d​⋅outputs_e,b,t,d​
        # For each feature index d, you’re averaging the experts’ outputs with a shared per-feature weight vector.

# standard 2-layer feed-forward network (MLP) with GELU activation and dropout
# “Per token, apply a 2-layer neural network to make the representation more expressive.”
# It doesn’t mix information across tokens (that’s what attention does); it only mixes features within each token vector.
class Mlp(nn.Module):
    # in_features: input dimension D_in (size of the last dimension of x).
    # hidden_features: inner dimension of the MLP (by default same as in_features).
    # out_features: output dimension (by default same as in_features).
    # act_layer: which activation to use (default: nn.GELU).
    # drop: dropout probability.
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # If out_features is None, it becomes in_features.
        out_features = out_features or in_features
        # If hidden_features is None, it also becomes in_features.
        hidden_features = hidden_features or in_features
        # Maps from size in_features to hidden_features.
        # If x has shape (B, N, D_in), then after fc1 it becomes (B, N, hidden_features).
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Creates the activation function instance.
        # Default: nn.GELU(), which is commonly used in Transformers.
        self.act = act_layer()
        # Maps from hidden_features back to out_features.
        # If x has shape (B, N, hidden_features), then after fc2 it becomes (B, N, out_features).
        # For standard Transformer FFN, often hidden_features = 4 * in_features, so it expands then compresses.
        self.fc2 = nn.Linear(hidden_features, out_features)
        # If drop > 0, randomly zeroes out some elements during training to regularize.
        self.drop = nn.Dropout(drop)

    # Let’s assume x shape is (B, N, D_in):
    def forward(self, x):
        # Applies linear transformation to the last dimension.
        # Shape: (B, N, hidden_features)
        x = self.fc1(x)
        # Applies non-linearity (GELU by default) element-wise.
        # Shape stays (B, N, hidden_features)
        x = self.act(x)
        # Applies dropout (if drop > 0) → randomly “drops” some activations.
        # Only active in training mode.
        # Shape unchanged.
        x = self.drop(x)
        # Second linear layer: maps back to out_features.
        # Shape becomes (B, N, out_features)
        x = self.fc2(x)
        # Dropout again on the final output (again only in training).
        # Shape unchanged.
        x = self.drop(x)
        return x

# x = feature map with shape (B, H, W, C) -> B: batch size, H: height, W: width, C: channels
# Split each image into non-overlapping windows of size window_size × window_size.
# Return all those windows stacked into one big batch: shape: (B * num_windows_per_image, window_size, window_size, C)
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # Split the H dimension into: H // window_size blocks (how many windows along height), each of size window_size.
    # Split the W dimension into: W // window_size blocks (how many windows along width), each of size window_size.
    # so the shape is: (B, num_windows_h, window_size, num_windows_w, window_size, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute dimensions to group windows nicely
    # The original dimension order after view is:
    # [B, num_windows_h, window_size_h, num_windows_w, window_size_w, C]
    #    0       1             2            3             4        5
    # flatten windows into one batch dimension, when we call .view(-1, window_size, window_size, C):
    # -1 means: “infer this dimension from the others”.
    # The first three dims (B, num_windows_h, num_windows_w) are flattened into one
    # So the final shape is:
    # (num_total_windows, window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# windows: all the windows after attention
# Goal: Put all those windows back into their correct positions and recover the full feature map: (B, H, W, C)
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # recover batch size B
    # windows.shape[0] is num_windows_total, i.e.
    # -> num_windows_total = B \times \frac{H}{window_size} \times \frac{W}{window_size}
    # H * W / window_size / window_size is:
    # -> \frac{H \times W}{window_size} \times {window_size} = \frac{H}{window_size} \times \frac{W}{window_size} = {num_windows_per_image}
    # So that formula is just algebra to get back the original B from how many windows there are.
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # reshape windows into a window grid
    # Now that we know B, we reshape the flat list of windows back into a grid of windows.
    # shape becomes: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    # First two dims after B (num_windows_h, num_windows_w) say which window in the grid (row, column).
    # Next two dims (window_size, window_size) are the pixels inside each window.
    # C is channels (left as -1, inferred from the original windows).
    # So now we have all windows back in a structured grid matching their layout in the original feature map.
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute axes to interleave windows into the image
    # Before permute:
    # (B, num_windows_h, num_windows_w, window_size, window_size, C)
    #  0       1             2            3            4        5
    # After permute:
    # (B, num_windows_h, window_size, num_windows_w, window_size, C)
    #  0       1             3            2            4        5
    # Why this reordering?
    # We want to combine num_windows_h and window_size into the full height H.
    # And combine num_windows_w and window_size into the full width W.
    # The new order groups them correctly as (B, H pieces, h_in_window, W pieces, w_in_window, C).
    # After permutation, the memory layout matches what we need for the final view
    # The .contiguous() ensures the tensor is laid out in memory in a way that .view() can safely reshape it.
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# self-attention inside windows
# adds a relative position bias: the model knows “this token is above-left of that one” etc
# supports shifted windows via an external mask (so some positions are masked out)
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        # dimension (size) of each attention head in a multi-head attention module.
        head_dim = dim // num_heads
        # self.scale: scaling factor for queries, typically 1 / sqrt(head_dim).
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # a relative offset Δh = i_h - j_h, Δw = i_w - j_w.
        # Δh ∈ [-(Wh-1), ..., 0, ..., (Wh-1)] → 2*Wh - 1 possibilities.
        # Δw ∈ [-(Ww-1), ..., 0, ..., (Ww-1)] → 2*Ww - 1 possibilities.
        # So total possible relative offsets = (2*Wh - 1) * (2*Ww - 1).
        # For each offset we have a bias per head → that’s the size of the table.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # Now they precompute an index that maps token pairs to those bias entries:
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords gives each position in the window its (row, col) coordinate.
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten flattens to a list of positions.
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords[..., i, j] = coords[:, i] - coords[:, j] → gives (Δh, Δw) between token i and j.
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # They map 2D offsets (Δh, Δw) into a single index using a kind of 2D → 1D encoding.
        # relative_position_index[i, j] is an integer in [0, (2*Wh-1)*(2*Ww-1)-1], pointing into relative_position_bias_table.
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # register_buffer stores this as a persistent, non-trainable tensor in the module.
        self.register_buffer("relative_position_index", relative_position_index)
        # This is all done once in __init__, so it’s cheap during forward.

        # qkv: single linear that predicts queries, keys, and values at once (3 * dim channels).
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # proj: final linear projection after attention.
        self.proj = nn.Linear(dim, dim)
        # attn_drop and proj_drop: dropout on attention weights and output.
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B_ = num_windows * B
        # N = Wh * Ww (number of tokens in each window).
        B_, N, C = x.shape
        # qkv = self.qkv(x)  # (B_, N, 3*C)
        # qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        # Shapes:
        # After qkv: (B_, N, 3*C)
        # After reshape: (B_, N, 3, num_heads, head_dim)
        # After permute: (3, B_, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # So:
        # q: (B_, num_heads, N, head_dim)
        # k: (B_, num_heads, N, head_dim)
        # v: (B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Scaled dot-product attention (before bias)
        # Multiply queries by 1 / sqrt(head_dim) for numerical stability.
        # Compute attention logits: q * k^T per head.
        q = q * self.scale
        #  # (B_, num_heads, N, N)
        # → For each batch-window and head, you have an N × N matrix of attention scores 
        # between all token pairs in that window.
        attn = (q @ k.transpose(-2, -1))

        # add relative position bias
        # self.relative_position_index is (N, N), each entry indicates which row in the bias table to use.
        # fter indexing and reshaping, relative_position_bias becomes (num_heads, N, N).
        # unsqueeze(0) → (1, num_heads, N, N), and then broadcasted to (B_, num_heads, N, N) and added to attn.
        # Effect:
        # Every pair of positions (i, j) in the window gets a bias depending on how far and in which direction j
        # is from i. This is what gives Swin Transformer its built-in notion of relative spatial layout.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # apply attention mask (for shifted windows)
        # When using shifted windows, some tokens in a window shouldn’t attend to others (because they
        # come from a different original window). This is encoded in mask.
        # mask has shape (num_windows, N, N) and typically contains 0 or -inf:
        # -> 0: allowed
        # -> -inf: disallowed (so after softmax that position gets zero probability).
        # B_ // nW recovers batch size.
        # They reshape attn so that window index lines up with mask, add the mask, flatten back, then apply softmax.
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        # If mask is None (non-shifted windows), they just softmax directly.
        else:
            attn = self.softmax(attn)

        # Now attn is: (B_, num_heads, N, N), with probabilities along the last dimension representing attention weights.

        # Attention → V, then projection
        attn = self.attn_drop(attn)

        # attn @ v:
        # -> attn: (B_, num_heads, N, N)
        # -> v: (B_, num_heads, N, head_dim)
        # -> result: (B_, num_heads, N, head_dim)
        # transpose(1, 2) → (B_, N, num_heads, head_dim)
        # reshape → (B_, N, C) (since C = num_heads * head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # self.proj(x) is a final linear mixing of heads back into the dim space.
        x = self.proj(x)
        # self.proj_drop applies dropout on the final output.
        x = self.proj_drop(x)
        # So the output shape is the same as input: (num_windows * B, N, C).
        return x

    # Just a pretty string for print(model) / debugging.
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    # Rough calculation of how many floating-point operations this layer uses for a window of length N.
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv projection
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        # # q @ k^T
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # x = (attn @ v)
        # attn @ v
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        # final proj
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    # dim: feature dimension (C).
    # input_resolution: (H, W) of the feature map for this block.
    # num_heads: attention heads used inside windows.
    # window_size: window height/width (e.g. 7).
    # shift_size: how much to shift the windows for shifted-window attention:
    # -> If 0 → normal window attention (W-MSA).
    # -> If >0 → shifted windows (SW-MSA).
    # mlp_ratio: how big the MLP hidden dimension is relative to dim (usually 4).
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # Handle tiny feature maps
        # If the feature map is smaller than the chosen window_size, you can’t split into that many windows.
        # Example: input_resolution = (4,4) and window_size = 7 → nonsense.
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            # Turn off shifting: shift_size = 0 (no SW-MSA).
            self.shift_size = 0
            # Set window_size to the smallest side of the feature map, e.g. 4 → one big window.
            self.window_size = min(self.input_resolution)
        # Sanity check: shift_size must be:
        # -> ≥ 0
        # -> strictly smaller than window_size.
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Layers: norm → attention → drop_path → norm → MLP
        # norm1: LayerNorm applied before attention (pre-norm style)
        self.norm1 = norm_layer(dim)
        # works on windows of size (self.window_size, self.window_size),
        # multi-head self-attention with relative position bias.
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath = stochastic depth: randomly drop the entire residual branch during training for regularization.
        # If drop_path == 0, they just use Identity (no effect).
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # norm2: LayerNorm before the MLP.
        self.norm2 = norm_layer(dim)
        # MLP with hidden dimension = mlp_ratio * dim.
        # mlp_hidden_dim: typically 4× dim (because mlp_ratio=4.).
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp: the 2-layer feed-forward network we discussed before:
        # -> dim → mlp_hidden_dim → dim.
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Building the attention mask for shifted windows
        # Only if we actually use shifted windows do we need a mask.
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            # img_mask is a fake “image” with 1 channel, full of zeros initially, shape (1, H, W, 1).
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            # Define 3 × 3 spatial regions (slices)
            # For height and width, they define 3 ranges each: (each slice)
            # This divides the feature map into a 3×3 grid of big blocks, something like:
            # -> First block: top-left region before last window_size rows/cols.
            # -> Second block: the “middle” bands.
            # -> Third block: bottom/right bands of size roughly shift_size.
            # This partitioning matches how windows wrap around when you shift by shift_size.
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            # Fill img_mask with region IDs
            # For each of the 3×3 blocks, they:
            # -> assign a unique integer cnt.
            # After this:
            # -> img_mask contains values 0,1,2,..., up to 8 (for 3×3 = 9 regions),
            # -> each pixel knows which big region it belongs to.
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Partition the mask into windows
            # window_partition(img_mask, window_size) cuts the “mask image” into the same windows that the real feature map will use.
            # Result shape: (num_windows, window_size, window_size, 1) (nW = number of windows per batch image.)
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            # Then we flatten each window’s (window_size, window_size) into a vector length N = window_size * window_size. -> mask_windows: (nW, N).
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # Compute pairwise differences → attention mask
            # mask_windows.unsqueeze(1) → (nW, 1, N)
            # mask_windows.unsqueeze(2) → (nW, N, 1)
            # If tokens i and j come from the same large region, this difference is 0.
            # If they come from different regions, the difference is non-zero.
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # Then they turn this into a real mask for attention:
            # -> Wherever attn_mask != 0 (different regions) → set to -100.0 (approx -inf).
            # -> Wherever attn_mask == 0 (same region) → set to 0.0.
            # Later, in WindowAttention.forward, they do:
            # -> attn = attn.view(B_ // nW, nW, num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # -> attn = softmax(attn)
            # -> Adding -100.0 before softmax makes those positions get probability ~0.
            # -> So tokens are not allowed to attend across certain region boundaries – that encodes the shifted-window pattern.
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # If there is no shift: They don’t need any mask, windows are “clean”.
        else:
            attn_mask = None

        # Stores attn_mask as a non-trainable tensor in the module.
        # It moves with .to(device) but is not updated by gradients.
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # self.input_resolution was passed in when the block was created, e.g. (H, W) = (56, 56)
        H, W = self.input_resolution
        # nput x has shape (B, L, C):
        # -> B: batch size
        # -> L: number of tokens (should be H * W)
        # -> C: channels (dim)
        B, L, C = x.shape
        # The assert just checks that L is consistent with H*W. If not, something upstream is wrong.
        assert L == H * W, "input feature has wrong size"

        # shortcut = x: save the original input for the residual connection later.
        shortcut = x
        # self.norm1(x): LayerNorm across C (per token) – this is “pre-norm” before attention.
        x = self.norm1(x)
        # view(B, H, W, C):
        # -> Reshape from (B, H*W, C) → (B, H, W, C)
        # -> So we see the features as a 2D feature map again.
        x = x.view(B, H, W, C)

        # cyclic shift
        # If shift_size == 0: no shift, this is regular window attention (W-MSA).
        # If shift_size > 0: shift the feature map cyclically:
        # -> torch.roll shifts along height (dim=1) and width (dim=2) by -shift_size.
        # -> “Cyclic” = things that exit one side re-enter on the opposite side.
        # This makes windows in this block misaligned compared to the previous block’s windows — that’s the
        # “shifted windows” idea in Swin.
        # We store the result as shifted_x (shape still (B, H, W, C)).
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # Cuts each (H, W) feature map into non-overlapping windows of size (window_size, window_size).
        # Output shape: (num_windows * B, window_size, window_size, C)
        # -> Let nW = number of windows per image → shape (B * nW, window_size, window_size, C).
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # Then we flatten each window’s (H, W) into a sequence:
        # Now shape is:
        # -> (B * nW, N, C), where N = window_size * window_size.
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # So each row in this batch is: “one window, seen as N tokens”.

        # W-MSA/SW-MSA
        # attn_mask is None if shift_size == 0 → normal windows.
        # Otherwise, it’s the mask we built in __init__ for shifted windows, so some tokens inside the window 
        # can’t see others (those that came from different original windows after shift).
        # Output attn_windows has the same shape as input: (B * nW, N, C).
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows back to feature map
        # First view: restore each window’s 2D shape: (B * nW, window_size, window_size, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # Inverse of window_partition.
        # It takes all windows and stitches them back to a (B, H, W, C) tensor.
        # H' and W' are actually just H and W here.
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # So shifted_x is now: same shape as x before partition: (B, H, W, C), but with updated features after attention.

        # reverse cyclic shift
        # If we had shifted earlier, we shift back by (shift_size, shift_size):
        # -> This brings tokens back to their original spatial locations.
        # -> Important: attention was computed in the shifted coordinate system; now we return to the normal one.
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # If there was no shift, just pass shifted_x through.
        else:
            x = shifted_x
        # view(B, H * W, C): We flatten back to the 2D “sequence of tokens” representation that the rest of the model uses.
        x = x.view(B, H * W, C)

        # FFN
        # Residual + MLP (feed-forward network)
        # First residual: attention branch
        # During training, sometimes scales x down and zeroes it (stochastic depth) for regularization
        # Add them: classic residual block.
        x = shortcut + self.drop_path(x)
        # Second residual: MLP branch
        # self.mpl -> (B, L, C) → (B, L, hidden_dim) → (B, L, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # PyTorch calls extra_repr() when you print a module (print(block) or print(model)).
    # It’s just a human-readable summary of key hyperparameters.
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    # This estimates how many floating-point operations (FLOPs) one forward pass of this block uses, for one image.
    def flops(self):
        flops = 0
        # Feature map size for this block.
        H, W = self.input_resolution
        # norm1
        # N_tokens = H * W.
        # LayerNorm touches each element of the feature map once:
        # -> There are H * W tokens.
        # -> Each token has self.dim features.
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        # self.attn.flops(N) returns FLOPs for one window with N tokens
        # Multiply by nW gives total FLOPs for all windows of this block: FLOPs_attn_block​=nW×FLOPs_attention_per_window​
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        # For the MLP we have two Linear layers:
        # -> 1st: dim → dim * mlp_ratio
        # -> 2nd: dim * mlp_ratio → dim
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

# downsampling layer between stages
# Think of it like “patch pooling + channel expansion”.
# Takes a feature map of size (H, W) with C channels.
# Merges 2×2 spatial patches into 1 token.
# So spatial resolution halves: (H, W) → (H/2, W/2).
# Channel count doubles: C → 2C.
# “Downsample spatially by 2, increase channels so information isn’t completely lost.”
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Linear layer that maps 4*C → 2*C.
        # This compresses the concatenated 4 tokens into a single token with 2× the original channel count.
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # We’ll concatenate 4 neighboring tokens’ channels → 4*C features per merged token.
        # So the normalization layer operates on vectors of size 4*C.
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # Input x is a flattened sequence: Shape (B, L, C) with L = H * W.
        H, W = self.input_resolution
        B, L, C = x.shape
        # Check that both H and W are even, because we need to take 2×2 patches.
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # Reshape back to image-like format: (B, H, W, C)
        x = x.view(B, H, W, C)

        # Take 2×2 patches and concatenate their channels
        # 0::2 means indices 0, 2, 4, ...
        # 1::2 means indices 1, 3, 5, ...
        # x0: pixels at (even row, even col)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1: pixels at (odd row, even col)
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2: pixels at (even row, odd col)
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3: pixels at (odd row, odd col)
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # Each of these has shape: B, H/2, W/2, C
        # Because we’re skipping every second row and column.
        # Visually, from each 2×2 block like:
        # (0,0)	(0,1)
        # (1,0) (1,1)
        # we’re extracting:
        # x0 → (0,0)
        # x1 → (1,0)
        # x2 → (0,1)
        # x3 → (1,1)

        # Concatenate along the channel dimension (-1).
        # So each 2×2 block becomes a single token whose feature vector is: [x0_channels,x1_channels,x2_channels,x3_channels]
        # New shape: (B, H/2, W/2, 4*C).
        # We just merged 4 neighboring tokens into 1, so:
        # -> Spatial resolution halved: H → H/2, W → W/2.\
        # -> Channel count quadrupled: C → 4*C.
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # Flatten spatial dimensions back into a sequence.
        # Now L_new = (H/2) * (W/2) = H*W / 4.
        # So shape is (B, L_new, 4*C).
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # LayerNorm on the last dimension of size 4*C.
        # Stabilizes and normalizes the concatenated features.
        x = self.norm(x)
        # nn.Linear(4*C → 2*C) applied to each token.
        # So final shape: (B, L_new, 2*C).
        x = self.reduction(x)

        # So the output has:
        # -> Spatial tokens: H/2 * W/2
        # -> Channels: 2*C
        # This matches the design: when you go down one stage in Swin:
        # -> Spatial resolution halves.
        # -> Channel count doubles.
        return x

    # Just for printing the module nicely.
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # Cost of normalization roughly proportional to number of elements (H * W * C).
        flops = H * W * self.dim
        # For the linear reduction:
        # -> Number of tokens after merging: (H/2) * (W/2)
        # -> Each token: 4 * dim input features → 2 * dim output features.
        # tokens×in_dim×out_dim=(H/2)(W/2)×(4⋅dim)×(2⋅dim)
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

# upsampling twin of PatchMerging
# it’s used in the decoder / expanding path.
class PatchExpand(nn.Module):
    # input_resolution: (H, W) of the current low-res feature map.
    # dim: number of channels of the input x (C).
    # dim_scale: Typically 2, Controls how much we downscale channels after expansion.
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # If dim_scale == 2:
        # -> Apply a linear layer to expand channels from C → 2C.
        # If not:
        # -> Just keep identity (no change).
        # In the usual Swin-Unet setting, dim_scale=2, so we expand channels to 2C first.
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        # After rearranging, the final per-token channel dimension will be C_out = dim // dim_scale.
        # -> For dim_scale=2, that’s C_out = C/2.
        # -> So LayerNorm runs on that final channel size.
        self.norm = norm_layer(dim // dim_scale)

    # Input x: (B, L, C) with L = H * W.
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        # If dim_scale == 2: C → 2C.
        # So now x is (B, H*W, 2C).
        x = self.expand(x)
        # Then we re-read the shape:
        B, L, C = x.shape
        # The assert just ensures that we’re consistent with H*W.
        assert L == H * W, "input feature has wrong size"

        # Reshape sequence back to spatial feature map: (B, H*W, C) → (B, H, W, C)
        x = x.view(B, H, W, C)

        # So far we just:
        # 1. Expanded channels (maybe),
        # 2. Restored 2D layout.

        # spatial upsampling
        # This uses einops.rearrange, which lets you reshape with a clear pattern.
        # Pattern: 'b h w (p1 p2 c) -> b (h p1) (w p2) c'
        # Before rearrange:
        # -> dims: (b, h, w, channels)
        # -> they require: channels = p1 * p2 * c
        # After rearrange:
        # -> dims: (b, h * p1, w * p2, c)
        # With the parameters:
        # -> p1 = 2, p2 = 2 → we are going to split the channels into 4 sub-groups that will become a 2×2 neighborhood.
        # c = C // 4 → that means: channels = p1 * p2 * c = 2 * 2 * (C // 4) = C, so this is consistent (we’re decomposing C into (2, 2, C//4)).
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        # Flatten (H*2, W*2) back to (L_out): L_out = (2H) * (2W) = 4 * H*W
        # Final shape: (B, L_out, C_out): C_out = C // 4 = dim // 2 (for dim_scale=2).
        x = x.view(B, -1, C // 4)
        # LayerNorm on the final channel dimension C_out.
        x = self.norm(x)

        return x

# it’s the very last upsampling layer
# “Take a low-res feature map and blow it up to the original input resolution (×4),
#  while getting back to the original channel dimension.”
class FinalPatchExpand_X4(nn.Module):
    # input_resolution: (H, W) for the current feature map.
    # dim: input channels C.
    # dim_scale=4: we’ll upsample H and W by factor 4.
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # Expands channels: C → 16C.
        # Why 16? Because for 4×4 spatial upsampling
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        # After rearranging and upsampling, each spatial position will have dim channels again.
        self.output_dim = dim
        # LayerNorm with feature size dim (the final number of channels).
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        # Linear layer expands channel dimension: dim → 16*dim.
        # Now x has shape: (B, H*W, 16*dim).
        x = self.expand(x)
        B, L, C = x.shape
        # The assert makes sure L still matches H * W.
        assert L == H * W, "input feature has wrong size"

        # Reshape back to image-like format: (B, H*W, C) → (B, H, W, C) Channels: C = 16*dim.
        x = x.view(B, H, W, C)
        # The 4×4 pixel-shuffle via rearrange
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        # Flatten (4H, 4W) into L_out = 16 * H * W.
        # Final shape: (B, 16 * H * W, dim).
        x = x.view(B, -1, self.output_dim)
        # LayerNorm along the channel dimension (dim).
        x = self.norm(x)

        # Output: (B, (4H * 4W), dim) → the final high-resolution feature map, 
        # ready to be turned into an image/mask by a 1×1 conv or similar.
        return x


# this is where all the pieces you’ve seen get stitched into a “stage” of Swin
# Contains several SwinTransformerBlocks stacked (depth times).
# Optionally ends with a downsampling (PatchMerging) to go to the next stage.
# Can optionally use checkpointing to save memory during training.
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    # dim: channels per token at this stage.
    # input_resolution: (H, W) for this stage.
    # depth: how many Swin blocks stacked in this stage.
    # num_heads, window_size, etc.: hyperparameters passed to each block.
    # downsample: module class to use at the end (e.g. PatchMerging) or None.
    # use_checkpoint: whether to use gradient checkpointing (trade extra compute for less memory).
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        # It creates a list of depth SwinTransformerBlocks.
        # shift_size alternates:
        # -> For even i (0, 2, 4, …): shift_size = 0 → W-MSA (non-shifted windows).
        # -> For odd i (1, 3, 5, …): shift_size = window_size // 2 → SW-MSA (shifted windows).
        # This is the Swin pattern: W-MSA → SW-MSA → W-MSA → SW-MSA → ...
        # -> drop_path handling:
        # -> -> If drop_path is a list (e.g. different stochastic depth rates per block), use drop_path[i] for that block.
        # -> -> Otherwise, if it’s just a single float, use the same rate for all blocks.
        # So self.blocks is like:
        # -> [Block(W-MSA), Block(SW-MSA), Block(W-MSA), Block(SW-MSA), ...] for depth layers.
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        # Optional downsampling at the end of the stage
        # If downsample is provided (e.g. PatchMerging class):
        # -> Instantiate it with this stage’s (input_resolution, dim).
        # -> This will be called after all blocks, to get: H, W → H/2, W/2, dim → 2*dim
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        # If downsample is None, then this stage does not change resolution (e.g. last stage).
        else:
            self.downsample = None

    def forward(self, x):
        # # Pass through all Swin blocks in this stage
        for blk in self.blocks:
            # If use_checkpoint is True:
            # -> Use torch.utils.checkpoint.checkpoint:
            # -> -> It recomputes each block’s forward during backprop instead of storing all intermediate activations.
            # -> -> Saves GPU memory at the cost of more compute.
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            # If use_checkpoint is False: Just do x = blk(x) normally.
            else:
                x = blk(x)
        # Optional downsample
        # If a PatchMerging was attached, apply it:
        # -> Spatial tokens go from (H*W) to (H/2 * W/2),
        # -> Channels go from dim to 2*dim.
        if self.downsample is not None:
            x = self.downsample(x)

        # So a whole stage is: x → Block1 → Block2 → ... → Block_depth → (optional) PatchMerging
        return x

    # For pretty printing.
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    # Sums the FLOPs of:
    # -> Each SwinTransformerBlock in self.blocks.
    # -> Plus the FLOPs of downsample (e.g. PatchMerging) if it exists.
    # This is used to estimate compute cost for that whole stage.
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# this is basically the decoder / upsampling twin of BasicLayer you just saw for the encoder.
# Same idea as BasicLayer: a stack of SwinTransformerBlocks at a fixed resolution.
# But instead of downsampling at the end with PatchMerging, it upsamples with PatchExpand.
# Used on the upsampling path of a Swin-UNet–style architecture.
# BasicLayer → encoder stage: blocks → (optional) PatchMerging
# BasicLayer_up → decoder stage: blocks → (optional) PatchExpand
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    # Same basic arguments as BasicLayer, except: upsample instead of downsample.
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        # Exactly the same pattern as in the encoder:
        # -> A stack of depth SwinTransformerBlocks.
        # -> shift_size alternates:
        # -> -> even index i: shift_size = 0 → W-MSA (non-shifted windows),
        # -> -> odd index i: shift_size = window_size // 2 → SW-MSA (shifted windows).
        # -> Each block gets its own drop_path if a list is provided, or the same value otherwise.
        # So within a stage (up or down), the Swin blocks behave the same: W-MSA ↔ SW-MSA alternating.
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        # Upsampling (PatchExpand)
        # If upsample argument is not None:
        # -> They create a PatchExpand layer with:
        # -> -> input_resolution: (H, W) at this stage
        # -> -> dim: current channels C
        # -> -> dim_scale=2: so it will double spatial resolution and halve channels
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        # If upsample is None, no upsampling is applied at the end of this stage.
        else:
            self.upsample = None

    def forward(self, x):
        # Run through all Swin blocks at this resolution
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # Optional upsampling
        # If we constructed PatchExpand, we apply it:
        # -> Spatial tokens: H*W → 4 * H*W (because 2× in both height and width).
        # -> Channels: C → C/2 (given dim_scale=2 usage).
        if self.upsample is not None:
            x = self.upsample(x)
        return x

# this is the very first step of Swin (and ViT-style) models: turn the image into a sequence of patch tokens
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution = how many patches along height and width: Ph = H_img / P_h, Pw = W_img / P_w
        # Example: img_size=224, patch_size=4: patches_resolution = [56, 56], num_patches = 56 * 56 = 3136.
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        # in_chans: number of input channels (3 for RGB, 1 for grayscale, etc.).
        self.in_chans = in_chans
        # embed_dim: dimension of each patch embedding.
        self.embed_dim = embed_dim

        # on an image of shape (B, C, H, W):
        # -> slides a patch-sized window across the image without overlap, stepping by the patch size.
        # -> Each kernel application “sees” one patch and outputs an embed_dim-dimensional vector.
        # So this acts like:
        # -> For each image patch of shape (C, P_h, P_w), apply a linear mapping → R^{embed_dim}.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Optional normalization
        # If you pass a norm layer (e.g. nn.LayerNorm), it creates self.norm.
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        # Otherwise no normalization on patches.
        else:
            self.norm = None

    def forward(self, x):
        # Expects input images to match exactly the configured img_size.
        # This implementation doesn’t support arbitrary sizes (you’d have to change this for that).
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # self.proj(x) -> Shape: (B, embed_dim, Ph, Pw), where Ph, Pw = self.patches_resolution.
        #  .flatten(2) -> Flattens the last two dims (Ph, Pw) into one: (B, embed_dim, Ph * Pw)
        #  .transpose(1, 2) -> Swap channel and sequence dims: (B, Ph * Pw, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # So now x is: Shape: (B, num_patches, embed_dim)
        # Exactly what a Transformer-like encoder expects: a batch of token sequences, each token a vector of size embed_dim.
        if self.norm is not None:
            x = self.norm(x)
        return x

    # This estimates FLOPs
    def flops(self):
        Ho, Wo = self.patches_resolution
        # For the Conv2d (i.e., patch projection):
        # -> Each output patch (Ho×Wo of them) computes: mbed_dim outputs, each output uses in_chans * (P_h * P_w) multiplications.
        # FLOPs_conv​≈Ho×Wo×embed_dim×in_chans×(P_h ​P_w​)
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        # For normalization: Roughly Ho * Wo * embed_dim operations.
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# this is the “big boss” that wires all the pieces you’ve been reading into a full Swin-UNet–style segmentation model
# I’ll walk through it in chunks:
# -> High-level idea
# -> Constructor arguments
# -> Encoder (self.layers)
# -> Decoder (self.layers_up + self.concat_back_dim)
# -> Final upsampling & output head
# -> Weight init
# High-level idea, swin is:
# -> A hierarchical encoder–decoder built using Swin Transformer blocks.
# -> Encoder: 4 stages that downsample and increase channels.
# -> Bottleneck at the deepest resolution.
# -> Decoder: 4 stages that upsample and fuse encoder features (via concatenation + linear).
# -> Final layer: upsample back to original patch grid and output segmentation logits.
# So: image → patches → Swin encoder → Swin decoder → segmentation map.
class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    # embed_dim: base channel dim after patch embedding.
    # depths: number of Swin blocks in each encoder stage.
    # depths_decoder: number of blocks in each decoder stage (not used directly here, interestingly).
    # num_heads: attention heads per stage.
    # window_size: Swin window size.
    # mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate: standard Transformer hyperparameters.
    # ape: use absolute positional embeddings or not.
    # patch_norm: whether to normalize patch embeddings.
    # use_checkpoint: memory/computation tradeoff.
    # final_upsample: strategy for final resolution step; here "expand_first" means use FinalPatchExpand_X4.
    def __init__(self, config=None, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        
        # Support config dict (codebase style) or individual parameters
        if config is not None:
            img_size = config.get("image_size", [224, 224])
            if isinstance(img_size, list):
                img_size = img_size[0]  # Use first dimension
            patch_size = config.get("patch_size", 4)
            in_chans = config.get("in_channels", 3)
            num_classes = config.get("out_channels", 1)
            embed_dim = config.get("embed_dim", 96)
            depths = config.get("depths", [2, 2, 2, 2])
            depths_decoder = config.get("depths_decoder", [1, 2, 2, 2])
            num_heads = config.get("num_heads", [3, 6, 12, 24])
            window_size = config.get("window_size", 7)
            mlp_ratio = config.get("mlp_ratio", 4.0)
            qkv_bias = config.get("qkv_bias", True)
            qk_scale = config.get("qk_scale", None)
            drop_rate = config.get("drop_rate", 0.0)
            attn_drop_rate = config.get("attn_drop_rate", 0.0)
            drop_path_rate = config.get("drop_path_rate", 0.1)
            ape = config.get("ape", False)
            patch_norm = config.get("patch_norm", True)
            use_checkpoint = config.get("use_checkpoint", False)
            final_upsample = config.get("final_upsample", "expand_first")
            
            # Store for final activation
            act_name = config.get("final_activation", None)
            if act_name == "sigmoid":
                self.final_activation = nn.Sigmoid()
            elif act_name == "softmax":
                self.final_activation = nn.Softmax(dim=1)
            else:
                self.final_activation = None
        else:
            self.final_activation = None

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # num_features = channels at the deepest encoder stage: embed_dim * 2^(num_layers-1), e.g. 96 * 2^3 = 768 for 4 stages.
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # num_features_up is used in some other implementations (not directly in the snippet), usually for decoder channels.
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        # If ape is True, create a learnable positional embedding of shape: (1, num_patches, embed_dim)
        # Initialized with truncated normal.
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr is a list of drop_path rates for each individual Swin block across all stages.
        # It linearly increases from 0 to drop_path_rate.
        # Later, segments of this list are assigned to each stage.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        # For each encoder stage i_layer: dim = embed_dim * 2^i_layer:
        # -> Stage 0: embed_dim
        # -> Stage 1: embed_dim*2
        # -> Stage 2: embed_dim*4
        # -> Stage 3: embed_dim*8
        # input_resolution decreases each stage:
        # -> patches_resolution / 2^i_layer
        # -> Because each previous stage applies PatchMerging and halves spatial size.
        # depth=depths[i_layer]: number of Swin blocks in this stage.
        # num_heads=num_heads[i_layer]: more heads in deeper stages.
        # drop_path= correct slice of dpr for this stage:
        # -> dpr[sum(depths[:i_layer]) : sum(depths[:i_layer+1])]
        # -> So each Swin block gets its own drop_path rate.
        # downsample=PatchMerging except for the last stage:
        # -> If i_layer < num_layers-1, add PatchMerging at the end of that stage.
        # -> Last stage has downsample=None (deepest resolution).
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        # linear layers to reduce channel dimension after concatenating skip connections.
        self.concat_back_dim = nn.ModuleList()
        # self.num_layers - 1 - i_layer counts backwards.
        # concat_linear logic:
        # -> For i_layer == 0 (deepest, no skip yet): Identity() (nothing to concat).
        # -> For later decoder stages, we’ll typically do:
        # -> concat([decoder_feat, encoder_skip]) along channels.
        # -> That doubles channels → dimension 2 * dim_current.
        # -> concat_linear maps 2 * dim_current → dim_current again.
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            # Next, building each decoder stage:
            # For the first decoder stage (i_layer == 0), they use just PatchExpand:
            # -> No Swin blocks here, just upsample from deepest resolution by ×2.
            # -> input_resolution & dim match the deepest encoder stage.
            # -> dim_scale=2 → 2× spatial and 0.5× channels.
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            # For subsequent decoder stages (i_layer > 0):
            # -> Use BasicLayer_up:
            # -> -> Contains Swin blocks at that resolution.
            # -> -> Optionally finishes with PatchExpand (if not the last decoder stage).
            # -> dim and input_resolution are chosen symmetrically to encoder:
            # -> -> Stage near the top uses smaller dim, bigger resolution.
            # -> depth and num_heads are taken from encoder depths and num_heads in reverse order.
            # -> drop_path slice again consistent with the corresponding encoder stage.
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            
            # layers_up[i] = upsampling stage i (either PatchExpand or BasicLayer_up).
            self.layers_up.append(layer_up)
            # -> concat_back_dim[i] = linear that reduces channel dim after concatenation at that stage.
            self.concat_back_dim.append(concat_linear)

        # self.norm: LayerNorm applied after encoder at deepest features. Feature dim = self.num_features (e.g. 768).
        self.norm = norm_layer(self.num_features)
        # self.norm_up: LayerNorm applied near the top of the decoder, where dim = embed_dim
        self.norm_up = norm_layer(self.embed_dim)

        # FinalPatchExpand_X4:
        # -> Takes features at patch resolution (H_patches, W_patches),
        # -> Upsamples them by ×4 in each dimension,
        # -> So you end up with full image resolution.
        # -> Output feature channels stay at embed_dim.
        # After FinalPatchExpand_X4, you get something like (B, embed_dim, H, W) (after reshaping from sequence).
        # self.output is a 1×1 convolution:
        # -> Maps embed_dim channels → num_classes.
        # -> For segmentation, this gives final logits per pixel.
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        # Weight initialization
        # Applies a custom _init_weights method (elsewhere in the class) to initialize all submodules:
        # -> Typically sets linear and conv weights with truncated normal
        # -> Norm layers with bias=0, weight=1, etc.
        self.apply(self._init_weights)

    # Weight initialization
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Initialize its weight with a truncated normal distribution
            trunc_normal_(m.weight, std=.02)
            # If it has a bias term
            if isinstance(m, nn.Linear) and m.bias is not None:
                # → bias starts at 0.
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Set bias to 0
            nn.init.constant_(m.bias, 0)
            # Set weight (scale) to 1
            nn.init.constant_(m.weight, 1.0)

    # Parameters excluded from weight decay
    # returns a set of parameter names that should not use weight decay
    # @torch.jit.ignore: When exporting / scripting the model with TorchScript, ignore these methods 
    # (they’re just helpers for optimizers, not needed in compiled graph).
    @torch.jit.ignore
    def no_weight_decay(self):
        # 'absolute_pos_embed' → positional embeddings are typically not regularized with weight decay.
        return {'absolute_pos_embed'}

    # returns a set of name substrings
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # Any parameter whose name contains 'relative_position_bias_table' should also be excluded from weight decay.
        # Relative position bias tables are learned offsets; you don’t usually want L2 decay on them.
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    # Input x: (B, C, H, W)
    def forward_features(self, x):
        # Output: (B, N_patches, embed_dim)
        # This cuts the image into patches and projects each patch into an embedding.
        x = self.patch_embed(x)
        if self.ape:
            # absolute_pos_embed: (1, N_patches, embed_dim)
            # Added element-wise across the patch dimension.
            x = x + self.absolute_pos_embed
        # Dropout on patches
        x = self.pos_drop(x)
        # Encoder stages with skip storage
        x_downsample = []

        # encoder stages
        # For each encoder stage:
        # -> First, store the current x in x_downsample (for skip connections later).
        # -> Then apply layer(x), which: Runs several Swin blocks, Optionally downsample with PatchMerging.
        # After this loop:
        # -> x is the deepest feature map (bottleneck).
        # x_downsample is a list of features at each encoder stage before downsampling, used for skips.
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        # Normalize deepest features
        # Applies LayerNorm along embedding dimension.
        # Shape remains (B, L, C) (L = final number of tokens, C = num_features).
        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        # Iterate over decoder stages
        # inx = decoder stage index (0, 1, 2, 3 for num_layers=4).
        # layer_up = either PatchExpand (for inx==0) or BasicLayer_up.
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    # Here, we go from patch grid back to full image resolution.
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # (B, H*W, C) → (B, (4H*4W), C') with C' = embed_dim, and internal spatial arrangement.
            x = self.up(x)
            # Reshape to image-like form
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            # Now height/width are 4H and 4W in pixels.
            # Channels = embed_dim.

            # Final 1×1 conv to class logits
            x = self.output(x)

        return x

    def forward(self, x):
        # Encoder + bottleneck: x: deepest features, x_downsample: skip features at all encoder stages.
        x, x_downsample = self.forward_features(x)
        # Decoder: Uses layers_up and concat_back_dim to upsample and fuse skips, Returns sequence at patch resolution. 
        x = self.forward_up_features(x, x_downsample)
        # Final 4× spatial upsampling to original image resolution.
        # 1×1 conv to num_classes.
        x = self.up_x4(x)
        
        # Apply final activation if configured
        if self.final_activation is not None:
            x = self.final_activation(x)

        # Return segmentation map.
        return x

    # FLOPs estimation
    def flops(self):
        flops = 0
        # patch embedding cost
        flops += self.patch_embed.flops()
        # Encoder stages
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        # Normalization at bottleneck -> Roughly: tokens at deepest resolution × channels.
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        # Final classification head
        flops += self.num_features * self.num_classes
        # Return total FLOPs
        return flops


# Register for codebase compatibility
@register_model('SwinTransformerSys')
class SwinTransformerSysWrapper(SwinTransformerSys):
    """Wrapper to make SwinTransformerSys compatible with codebase config system."""
    def __init__(self, config: dict):
        # Call parent with config as first argument
        super().__init__(config=config)
