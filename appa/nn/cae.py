r"""Conv Auto-Encoder (CAE) building blocks."""

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAE",
    "conv_ae",
]

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Sequence, Tuple, Union

from appa.nn import triggers

from .layers import (
    ConvNd,
    LayerNorm,
    Patchify,
    Unpatchify,
)


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module.

    Arguments:
        channels: The number of channels :math:`C`.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Norm
        self.norm = LayerNorm(dim=-spatial - 1)

        # FFN
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        self.ffn[-1].weight.data.mul_(1e-2)
        if self.ffn[-1].bias is not None:
            self.ffn[-1].bias.data.zero_()

    def _forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = self.norm(x)
        y = self.ffn(y)

        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class ConvEncoder(nn.Module):
    r"""Creates an conv encoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        periodic_lon: Whether only the longitude dimension is periodic.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        periodic_lon: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if periodic_lon:
            assert spatial == 2, "Longitude-only periodic padding requires 2 spatial dimensions."

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode=("constant", "circular")
            if periodic_lon and not periodic
            else "circular"
            if periodic
            else "zeros",
        )

        self.patch = Patchify(patch_size=patch_size)
        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 / 2^D, ..., L_N  / 2^D)`.
        """

        x = self.patch(x)

        for blocks in self.descent:
            for block in blocks:
                x = block(x)

        return x


class ConvDecoder(nn.Module):
    r"""Creates a conv decoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        periodic_lon: Whether only the longitude dimension is periodic.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        periodic_lon: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if periodic_lon:
            assert spatial == 2, "Longitude-only periodic padding requires 2 spatial dimensions."

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode=("constant", "circular")
            if periodic_lon and not periodic
            else "circular"
            if periodic
            else "zeros",
        )

        self.unpatch = Unpatchify(patch_size=patch_size)
        self.ascent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                            Unpatchify(patch_size=stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 \times 2^D, ..., L_N  \times 2^D)`.
        """

        for blocks in self.ascent:
            for block in blocks:
                x = block(x)

        x = self.unpatch(x)

        return x


class ConvAE(nn.Module):
    r"""Creates a conv auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        shape: The input shape.
        resize: The resize factor.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        shape: Sequence[int] = (721, 1440),
        resize: int = 32,
        saturation: str = "softclip2",
        saturation_bound: float = 5.0,
        noise: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.shape = shape
        self.resize = resize

        self.saturation = saturation
        self.saturation_bound = saturation_bound
        self.noise = noise

    def description(self):
        # TO IMPROVE
        return "Conv autoencoder"

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / self.saturation_bound)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / self.saturation_bound))
        elif self.saturation == "tanh":
            return torch.tanh(x / self.saturation_bound) * self.saturation_bound
        elif self.saturation == "asinh":
            return torch.arcsinh(x)
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def pad(self, x: Tensor) -> Tensor:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f

        x = torch.nn.functional.pad(x, pad=(w // 2, w - w // 2, 0, 0), mode="circular")
        x = torch.nn.functional.pad(x, pad=(0, 0, h // 2, h - h // 2), mode="constant")

        return x

    def unpad(self, x: Tensor) -> Tensor:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f

        if h > 0:
            x = x[..., h // 2 : h // 2 - h, :]

        if w > 0:
            x = x[..., :, w // 2 : w // 2 - w]

        return x

    @property
    def latent_shape(self) -> int:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f
        h, w = (H + h) // f, (W + w) // f

        latent_channels = self.encoder.descent[-1][-1].out_channels

        return h, w, latent_channels

    def encode(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        H, W = self.shape
        x = rearrange(x, "... (H W) C -> ... C H W", H=H, W=W)
        x = self.pad(x)
        z = self.encoder(x)
        z = self.saturate(z)
        z = rearrange(z, "... C H W -> ... (H W) C")
        return z

    def decode(self, z: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        H, W, _ = self.latent_shape
        if self.noise > 0:
            z = z + self.noise * torch.randn_like(z)
        z = rearrange(z, "... (H W) C -> ... C H W", H=H, W=W)
        x = self.decoder(z)
        x = self.unpad(x)
        x = rearrange(x, "... C H W -> ... (H W) C")
        return x

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, t, c)
        y = self.decode(z, t, c)
        return z, y


def conv_ae(
    in_channels: int,
    lat_channels: int,
    spatial: int = 2,
    # AE
    shape: Sequence[int] = (721, 1440),
    resize: int = 32,
    saturation: str = "softclip2",
    saturation_bound: float = 5.0,
    noise_level: float = 0.0,
    periodic_lon: bool = False,
    # Ignore
    latent_shape: Optional[Tuple] = None,
    name: Optional[str] = None,
    context_channels: Optional[int] = None,
    # Passthrough
    **kwargs,
) -> ConvAE:
    r"""Instantiates a conv auto-encoder."""

    encoder = ConvEncoder(
        in_channels=in_channels,
        out_channels=lat_channels,
        spatial=spatial,
        periodic_lon=periodic_lon,
        **kwargs,
    )

    decoder = ConvDecoder(
        in_channels=lat_channels,
        out_channels=in_channels,
        spatial=spatial,
        periodic_lon=periodic_lon,
        **kwargs,
    )

    autoencoder = ConvAE(
        encoder,
        decoder,
        shape=shape,
        resize=resize,
        saturation=saturation,
        saturation_bound=saturation_bound,
        noise=noise_level,
    )

    return autoencoder
