r"""Low-resolution adapter around a pretrained convolutional autoencoder."""

import torch.nn as nn
import torch.nn.functional as F

from appa.nn.layers import LayerNorm
from einops import rearrange


TARGET_CHANNELS = 71


class AdapterResBlock(nn.Module):
    r"""Residual refinement block for low-resolution adapters."""

    def __init__(self, channels: int):
        super().__init__()

        self.net = nn.Sequential(
            LayerNorm(dim=-3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.net[-1].weight.data.mul_(1e-2)
        if self.net[-1].bias is not None:
            self.net[-1].bias.data.zero_()

    def forward(self, x):
        return x + self.net(x)


def adapter_resblocks(channels: int, blocks: int) -> list[nn.Module]:
    return [AdapterResBlock(channels) for _ in range(blocks)]


def build_input_adapter(
    input_channels: int,
    injection_shape: tuple[int, int],
    block_type: str,
    blocks_per_stage: int,
) -> nn.Sequential:
    if block_type == "plain":
        return nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(injection_shape),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    if block_type == "resnet":
        return nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            *adapter_resblocks(128, blocks_per_stage),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            *adapter_resblocks(256, blocks_per_stage),
            nn.AdaptiveAvgPool2d(injection_shape),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            *adapter_resblocks(512, blocks_per_stage),
        )

    raise ValueError(f"Unknown adapter block_type: {block_type}")


class OutputAdapter(nn.Module):
    r"""Map the reused decoder feature map back to the low-resolution field."""

    def __init__(
        self,
        input_channels: int = 512,
        output_channels: int = TARGET_CHANNELS,
        output_shape: tuple[int, int] = (121, 240),
        block_type: str = "plain",
        blocks_per_stage: int = 1,
    ):
        super().__init__()

        self.output_shape = tuple(output_shape)
        self.block_type = block_type

        if block_type == "plain":
            self.proj = nn.Sequential(
                nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            )
        elif block_type == "resnet":
            self.proj = nn.Sequential(
                nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
                nn.SiLU(),
                *adapter_resblocks(256, blocks_per_stage),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.SiLU(),
                *adapter_resblocks(128, blocks_per_stage),
                nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown adapter block_type: {block_type}")

    def forward(self, x):
        if self.block_type == "resnet":
            x = F.interpolate(x, size=self.output_shape, mode="bilinear", align_corners=False)
            return self.proj(x)

        x = self.proj(x)
        return F.interpolate(x, size=self.output_shape, mode="bilinear", align_corners=False)


class LowResAdaptedConvAE(nn.Module):
    r"""Adapter model that reuses later encoder and early decoder checkpoint blocks."""

    def __init__(
        self,
        ae,
        lowres_shape: tuple[int, int] = (121, 240),
        injection_shape: tuple[int, int] = (92, 188),
        freeze_reused: bool = True,
        input_channels: int = TARGET_CHANNELS,
        block_type: str = "plain",
        blocks_per_stage: int = 1,
    ):
        super().__init__()

        self.lowres_shape = tuple(lowres_shape)
        self.injection_shape = tuple(injection_shape)
        self.input_channels = input_channels
        self.block_type = block_type
        self.saturation = ae.saturation
        self.saturation_bound = ae.saturation_bound
        self.saturate = ae.saturate

        self.input_adapter = build_input_adapter(
            input_channels=input_channels,
            injection_shape=self.injection_shape,
            block_type=block_type,
            blocks_per_stage=blocks_per_stage,
        )
        self.encoder_stage3_resblocks = nn.ModuleList(ae.encoder.descent[3][1:])
        self.encoder_blocks = nn.ModuleList(ae.encoder.descent[4:])
        self.decoder_blocks = nn.ModuleList(ae.decoder.ascent[:2])
        self.output_adapter = OutputAdapter(
            input_channels=512,
            output_channels=input_channels,
            output_shape=self.lowres_shape,
            block_type=block_type,
            blocks_per_stage=blocks_per_stage,
        )

        if freeze_reused:
            self.freeze_reused()

    @property
    def latent_shape(self):
        return 23, 47, 128

    def description(self):
        return f"Low-resolution adapted Conv autoencoder ({self.block_type} adapters)"

    def freeze_reused(self) -> None:
        for module in (self.encoder_stage3_resblocks, self.encoder_blocks, self.decoder_blocks):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def encode_image(self, x, shapes=None):
        if shapes is not None:
            shapes.append(("input", tuple(x.shape)))

        x = self.input_adapter(x)
        if shapes is not None:
            shapes.append(("input_adapter", tuple(x.shape)))

        for block in self.encoder_stage3_resblocks:
            x = block(x)
        if shapes is not None:
            shapes.append(("encoder.descent[3][1:]", tuple(x.shape)))

        for i, blocks in enumerate(self.encoder_blocks, start=4):
            for block in blocks:
                x = block(x)
            if shapes is not None:
                shapes.append((f"encoder.descent[{i}]", tuple(x.shape)))

        z = self.saturate(x)
        if shapes is not None:
            shapes.append(("latent", tuple(z.shape)))

        return z

    def decode_image(self, z, shapes=None):
        x = z
        for i, blocks in enumerate(self.decoder_blocks):
            for block in blocks:
                x = block(x)
            if shapes is not None:
                shapes.append((f"decoder.ascent[{i}]", tuple(x.shape)))

        x = self.output_adapter(x)
        if shapes is not None:
            shapes.append(("output_adapter", tuple(x.shape)))

        return x

    def forward_image(self, x, return_shapes=False):
        shapes = [] if return_shapes else None
        z = self.encode_image(x, shapes=shapes)
        y = self.decode_image(z, shapes=shapes)

        if return_shapes:
            return z, y, shapes
        return z, y

    def forward(self, x, t=None, c=None, return_shapes=False):
        if x.ndim == 4:
            return self.forward_image(x, return_shapes=return_shapes)

        H, W = self.lowres_shape
        x_img = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        z_img, y_img = self.forward_image(x_img, return_shapes=False)
        z = rearrange(z_img, "B C H W -> B (H W) C")
        y = rearrange(y_img, "B C H W -> B (H W) C")

        return z, y
