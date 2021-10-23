from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

import pystiche

from ..utils import AutoPadConv2d, AutoPadConvTranspose2d, ResidualBlock
from pystiche_papers.bueltemeier_mst_2021._modules import MSTSequentialEncoder

__all__ = [
    "conv",
    "conv_block",
    "residual_block",
    "encoder",
    "decoder",
]


def conv(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        padding: Optional[Union[Tuple[int, int], int]] = None,
        upsample: bool = False,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    cls: Union[Type[nn.Conv2d], Type[nn.ConvTranspose2d]]
    kwargs: Dict[str, Any]
    if padding is None:
        cls = AutoPadConvTranspose2d if upsample else AutoPadConv2d
        kwargs = {}
    else:
        cls = nn.ConvTranspose2d if upsample else nn.Conv2d
        kwargs = dict(padding=padding)
    return cls(in_channels, out_channels, kernel_size, stride=stride, **kwargs)


def conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        padding: Optional[Union[Tuple[int, int], int]] = None,
        upsample: bool = False,
        relu: bool = True,
        inplace: bool = True,
) -> nn.Sequential:
    modules: List[nn.Module] = [
        conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
        ),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
    ]
    if relu:
        modules.append(nn.ReLU(inplace=inplace))
    return nn.Sequential(*modules)


def residual_block(channels: int, inplace: bool = True) -> ResidualBlock:
    in_channels = out_channels = channels
    kernel_size = 3
    residual = nn.Sequential(
        conv_block(in_channels, out_channels, kernel_size, stride=1, inplace=inplace,),
        conv_block(in_channels, out_channels, kernel_size, stride=1, relu=False,),
    )

    return ResidualBlock(residual)


def encoder(in_channels=3) -> MSTSequentialEncoder:
    modules = (
        conv_block(in_channels=in_channels, out_channels=32, kernel_size=9,),
        conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2,),
        conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2,),
    )
    return MSTSequentialEncoder(modules)


def bottleneck(in_channels, n_blocks=5):
    modules = []
    for i in range(n_blocks):
        modules.append(residual_block(channels=in_channels))
    return nn.Sequential(*modules)


def decoder(in_channels, out_channels=3) -> pystiche.SequentialModule:
    class ValueRangeDelimiter(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x)

    modules = (
        conv_block(
            in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, upsample=True,
        ),
        conv_block(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, upsample=True,
        ),
        AutoPadConv2d(in_channels=32, out_channels=out_channels, kernel_size=9,),
        ValueRangeDelimiter(),
    )

    return pystiche.SequentialModule(*modules)
