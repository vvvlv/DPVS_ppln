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

from ..registry import register_model
from ..blocks.swin_blocks import LinearEmbedding, TinySwinTransformerBlock, PatchMerging, PatchExpand

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
        dropout_rate = config.get("dropout", 0.0)
        use_conv_stem = config.get("conv_stem", False)
        conv_stem_kernel_size = config.get("conv_stem_kernel_size", 3)
        conv_stem_layers = config.get("conv_stem_layers", 1)

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

        # Optional convolutional stem to denoise / normalize local contrast
        # before tokenization. Useful when vessels have very different scales
        # and contrast.
        # Configurable kernel size and number of layers:
        # - kernel_size: 3 (preserves fine vessels 1-2px), 5 (better noise reduction, may blur fine vessels)
        # - layers: 1 (standard), 2+ (better noise reduction while preserving details)
        if use_conv_stem:
            if conv_stem_kernel_size % 2 == 0:
                raise ValueError(f"conv_stem_kernel_size must be odd, got {conv_stem_kernel_size}")
            if conv_stem_layers < 1:
                raise ValueError(f"conv_stem_layers must be >= 1, got {conv_stem_layers}")
            
            padding = conv_stem_kernel_size // 2
            stem_layers = []
            for _ in range(conv_stem_layers):
                stem_layers.extend([
                    nn.Conv2d(in_channels, in_channels, kernel_size=conv_stem_kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                ])
            self.conv_stem = nn.Sequential(*stem_layers)
        else:
            self.conv_stem = None

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
                        dropout_rate=dropout_rate,
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
                        dropout_rate=dropout_rate,
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
        # Optional conv stem for noise reduction / local contrast adaptation
        if self.conv_stem is not None:
            x = self.conv_stem(x)

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


            


        