import pytest
import torch

from pathlib import Path

from appa.grid import create_icosphere, create_N320
from appa.nn.cae import ResBlock, conv_ae
from appa.nn.gae import GraphAE
from appa.nn.graph import GraphPoolAttention, GraphSelfAttention
from appa.nn.layers import ConvNd

param_combinations = [
    (1, 3, 1, [16], 2, [2], [2], [1.5], "softclip", 0.5),
    (1, 4, 2, [8], 2, [2], [2], [1.5], "softclip2", 0.5),
    (1, 3, 1, [8], 2, [2], [2], [1.5], "tanh", 0.5),
    (1, 3, 2, [8], 2, [2], [2], [2], "asinh", 0.5),
    (1, 3, 1, [8], 2, [2], [2], [3], None, 0.5),
]


def test_convnd_identity_init_zeros_bias():
    conv = ConvNd(
        in_channels=2,
        out_channels=4,
        spatial=2,
        kernel_size=3,
        padding=1,
        identity_init=True,
    )

    assert conv.bias is not None
    assert torch.all(conv.bias == 0)


def test_resblock_final_bias_is_zero():
    block = ResBlock(channels=4, kernel_size=3, padding=1)

    assert block.ffn[-1].bias is not None
    assert torch.all(block.ffn[-1].bias == 0)


@pytest.mark.parametrize(
    "batch_size, in_channels, context_channels, hid_channels, latent_channels, ico_divisions, self_attention_blocks, pooling_area, saturation, saturation_bound",
    param_combinations,
)
def test_gae(
    tmp_path: Path,
    batch_size: int,
    in_channels: int,
    context_channels: int,
    hid_channels: list,
    latent_channels: int,
    ico_divisions: list,
    self_attention_blocks: int,
    pooling_area: list,
    saturation: str,
    saturation_bound: float,
):
    autoencoder = GraphAE(
        in_channels=in_channels,
        context_channels=context_channels,
        hid_channels=hid_channels,
        heads=[2] * len(hid_channels),
        latent_channels=latent_channels,
        ico_divisions=ico_divisions,
        self_attention_blocks=self_attention_blocks,
        pooling_area=pooling_area,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )

    N_lat = 721
    N_lon = 1440

    x = torch.randn(size=(batch_size, N_lat * N_lon, in_channels))
    c = torch.randn(size=(batch_size, N_lat * N_lon, context_channels))
    # random date, at most 12 for month
    t = torch.randint(13, size=(batch_size, 4))
    z, x_reconstructed = autoencoder(x, t, c)

    # Saturation
    if saturation not in [None, "asinh"]:
        assert torch.all(z.abs() <= saturation_bound)

    # Shapes
    iso_div_count = sum(10 * 2 ** (ico_div * 2) + 2 for ico_div in ico_divisions)
    assert z.shape == (batch_size, iso_div_count, latent_channels)
    assert x_reconstructed.shape == (batch_size, N_lat * N_lon, in_channels)

    # Gradients
    assert z.requires_grad
    assert x_reconstructed.requires_grad

    loss = x_reconstructed.square().sum()
    loss.backward()

    for p in autoencoder.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(autoencoder.state_dict(), tmp_path / "autoencoder_state.pth")

    del autoencoder, loss

    # Load
    autoencoder_copy = GraphAE(
        in_channels=in_channels,
        context_channels=context_channels,
        hid_channels=hid_channels,
        heads=[2] * len(hid_channels),
        latent_channels=latent_channels,
        ico_divisions=ico_divisions,
        self_attention_blocks=self_attention_blocks,
        pooling_area=pooling_area,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )
    autoencoder_copy.load_state_dict(
        torch.load(tmp_path / "autoencoder_state.pth", weights_only=True)
    )

    autoencoder_copy.eval()

    z_copy, x_reconstructed_copy = autoencoder_copy(x, t, c)

    assert torch.allclose(z, z_copy)
    assert torch.allclose(x_reconstructed, x_reconstructed_copy)


param_combinations = [
    (2, [16], 2, [4], [2], [2], [3], True),
    (2, [8, 8], 2, [4, 3], [1, 1], [2, 3], [3, 3], True),
    (2, [8, 8, 16], 2, [4, 3, 2], [1, 1, 1], None, [3, 3, 3], False),
]


@pytest.mark.parametrize(
    "in_channels, hid_channels, latent_channels, ico_divisions, self_attention_blocks, self_attention_hops, pooling_area, use_hop_pooling",
    param_combinations,
)
def test_gae_arch(
    in_channels: int,
    hid_channels: list,
    latent_channels: int,
    ico_divisions: list,
    self_attention_blocks: int,
    self_attention_hops: list,
    pooling_area: list,
    use_hop_pooling: bool,
):
    autoencoder = GraphAE(
        in_channels=in_channels,
        context_channels=0,
        hid_channels=hid_channels,
        heads=[2] * len(hid_channels),
        latent_channels=latent_channels,
        ico_divisions=ico_divisions,
        self_attention_blocks=self_attention_blocks,
        pooling_area=pooling_area,
        use_hop_pooling=use_hop_pooling,
        self_attention_hops=self_attention_hops,
    )

    icosphere_graphs = [create_icosphere(N) for N in ico_divisions]
    grids = [create_N320(), *[vertices for vertices, edges in icosphere_graphs]]
    grid_N_nodes = [grid.shape[0] for grid in grids]

    key_idx = 0
    key_increment = 1
    for component in [autoencoder.encoder, autoencoder.decoder]:
        for layer in component:
            if isinstance(layer, (GraphSelfAttention, GraphPoolAttention)):
                # all nodes should be connected at least once
                if isinstance(layer, GraphPoolAttention):
                    # pooling blocks goes to the next grid
                    query_set = layer.edges[:, 0].unique().sort()[0]
                    query_target_set = torch.arange(grid_N_nodes[key_idx + key_increment])

                    key_set = layer.edges[:, 1].unique().sort()[0]
                    key_target_set = torch.arange(grid_N_nodes[key_idx])

                    key_idx += key_increment
                else:
                    query_set = layer.edges[:, 0].unique().sort()[0]
                    query_target_set = torch.arange(grid_N_nodes[key_idx])

                    key_set = layer.edges[:, 1].unique().sort()[0]
                    key_target_set = torch.arange(grid_N_nodes[key_idx])

                assert (
                    query_set.shape == query_target_set.shape
                ), f"Shape is not the same. Target: {query_target_set.shape}, got: {query_set.shape}"
                assert (
                    key_set.shape == key_target_set.shape
                ), f"Shape is not the same. Target: {key_target_set.shape}, got: {key_set.shape}"

                # Assert that the nodes are the same (even if we have the same number of them)
                assert torch.all(
                    query_set == query_target_set
                ), f"Some query nodes are not connected. Target: {query_target_set.shape}, got: {query_set.shape}"
                assert torch.all(
                    key_set == key_target_set
                ), f"Some key nodes are not connected. Target: {key_target_set.shape}, got: {key_set.shape}"

        # We switch to the decoder and then go back in the grids list
        key_increment = -1


param_combinations = [
    (1, 3, 1, [16], 2, "softclip", 0.5),
    (1, 4, 2, [8], 2, "softclip2", 0.5),
    (1, 3, 1, [8], 2, "tanh", 0.5),
    (1, 3, 2, [8], 2, "asinh", 0.5),
    (1, 3, 1, [8], 2, None, 0.5),
]


@pytest.mark.parametrize(
    "batch_size, in_channels, context_channels, hid_channels, latent_channels, saturation, saturation_bound",
    param_combinations,
)
def test_cae(
    tmp_path: Path,
    batch_size: int,
    in_channels: int,
    context_channels: int,
    hid_channels: list,
    latent_channels: int,
    saturation: str,
    saturation_bound: float,
):
    autoencoder = conv_ae(
        in_channels=in_channels,
        context_channels=context_channels,
        hid_channels=hid_channels,
        hid_blocks=[1] * len(hid_channels),
        resize=2 ** (len(hid_channels) - 1),
        lat_channels=latent_channels,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )

    N_lat = 721
    N_lon = 1440

    x = torch.randn(size=(batch_size, N_lat * N_lon, in_channels))
    c = torch.randn(size=(batch_size, N_lat * N_lon, context_channels))
    # random date, at most 12 for month
    t = torch.randint(13, size=(batch_size, 4))
    z, x_reconstructed = autoencoder(x, t, c)

    # Saturation
    if saturation not in [None, "asinh"]:
        assert torch.all(z.abs() <= saturation_bound)

    # Shapes
    h, w, _ = autoencoder.latent_shape
    assert z.shape == (batch_size, h * w, latent_channels)
    assert x_reconstructed.shape == (batch_size, N_lat * N_lon, in_channels)

    # Gradients
    assert z.requires_grad
    assert x_reconstructed.requires_grad

    loss = x_reconstructed.square().sum()
    loss.backward()

    for p in autoencoder.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(autoencoder.state_dict(), tmp_path / "autoencoder_state.pth")

    del autoencoder, loss

    # Load
    autoencoder_copy = conv_ae(
        in_channels=in_channels,
        context_channels=context_channels,
        hid_channels=hid_channels,
        hid_blocks=[1] * len(hid_channels),
        resize=2 ** (len(hid_channels) - 1),
        lat_channels=latent_channels,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )
    autoencoder_copy.load_state_dict(
        torch.load(tmp_path / "autoencoder_state.pth", weights_only=True)
    )

    autoencoder_copy.eval()

    z_copy, x_reconstructed_copy = autoencoder_copy(x, t, c)

    assert torch.allclose(z, z_copy)
    assert torch.allclose(x_reconstructed, x_reconstructed_copy)
