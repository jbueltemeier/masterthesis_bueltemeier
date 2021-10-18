from typing import Any, Dict, Union, Tuple, List, Optional, Type, cast

from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ..utils import AutoPadConv2d

from pystiche import enc, core, meta


__all__ = [
    "SequentialDecoder",
    "norm",
    "conv",
    "upsample_block",
    "conv_block",
    "ResidualBlock",
    "UpResidualBlock",
    "encoder",
    "Inspiration",
    "bottleneck",
    "decoder",
]


class MSTSequentialEncoder(enc.SequentialEncoder):
    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        # for module in self.children():
        #     if isinstance(module, ResidualBlock) or isinstance(module, nn.Sequential):
        #         # TODO: current workaround, implement this properly (conv handling)
        #         guide = cast(torch.Tensor, F.max_pool2d(guide, kernel_size=2))
        #     else:
        #         guide = enc.guides.propagate_guide(module, guide)
        guide = cast(torch.Tensor, F.max_pool2d(guide, kernel_size=2))
        guide = cast(torch.Tensor, F.max_pool2d(guide, kernel_size=2))
        return guide


class SequentialDecoder(core.SequentialModule):
    def __init__(self, *modules: nn.Module):
        super().__init__(*modules)


def norm(
    out_channels: int, instance_norm: bool
) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    norm_kwargs: Dict[str, Any] = {
        "eps": 1e-5,
        "momentum": 1e-1,
        "affine": True,
        "track_running_stats": True,
    }
    if instance_norm:
        return nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        return nn.BatchNorm2d(out_channels, **norm_kwargs)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    cls: Type[nn.Conv2d]
    kwargs: Dict[str, Any]
    if padding is None:
        cls = AutoPadConv2d
        kwargs = dict(padding_mode="reflect")
    else:
        cls = nn.Conv2d
        kwargs = dict(padding=padding)
    return cls(in_channels, out_channels, kernel_size, stride=stride, **kwargs)


def upsample_block(scale_factor=2.0) -> nn.Upsample:
    r"""Upsample the input to scale the size using an :class:`~torch.nn.Upsample`."""
    return nn.Upsample(scale_factor=scale_factor, mode="nearest")


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int] = 3,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: Optional[int] = None,
    inplace: bool = True,
    instance_norm: bool = False,
) -> nn.Sequential:
    modules: List[nn.Module] = [
        norm(in_channels, instance_norm),
        nn.ReLU(inplace=inplace),
    ]
    if upsample:
        modules += [upsample_block(scale_factor=upsample)]

    modules += [
        conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    ]
    return nn.Sequential(*modules)


class ResidualBlock(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        expansion=4,
        downsample=None,
        instance_norm=False,
    ):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(
                in_channels, channels * expansion, kernel_size=1, stride=stride
            )

        modules = [
            conv_block(
                in_channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                instance_norm=instance_norm,
            ),
            conv_block(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                instance_norm=instance_norm,
            ),
            conv_block(
                channels,
                channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                instance_norm=instance_norm,
            ),
        ]

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        residual = self.residual_layer(x) if self.downsample is not None else x
        return residual + self.conv_block(x)


class UpResidualBlock(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(
        self, in_channels, channels, stride=2, expansion=4, instance_norm=False
    ):
        super(UpResidualBlock, self).__init__()
        self.residual_layer = conv_block(
            in_channels,
            channels * expansion,
            kernel_size=1,
            stride=1,
            upsample=stride,
            instance_norm=instance_norm,
        )

        modules = [
            conv_block(
                in_channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                instance_norm=instance_norm,
            ),
            conv_block(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                upsample=stride,
                instance_norm=instance_norm,
            ),
            conv_block(
                channels,
                channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                instance_norm=instance_norm,
            ),
        ]

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


def encoder(in_channels=3, channels=64, expansion=4, instance_norm=False):
    modules = [
        conv(in_channels, channels, kernel_size=7, stride=1),
        norm(channels, instance_norm=instance_norm),
        nn.ReLU(inplace=True),
        ResidualBlock(
            channels,
            32,
            stride=2,
            expansion=expansion,
            downsample=1,
            instance_norm=instance_norm,
        ),
        ResidualBlock(
            32 * expansion,
            channels,
            stride=2,
            expansion=expansion,
            downsample=1,
            instance_norm=instance_norm,
        ),
    ]
    return MSTSequentialEncoder(modules)


class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(
            self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
            X.view(X.size(0), X.size(1), -1),
        ).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "N x " + str(self.C) + ")"


def bottleneck(in_channels, expansion=4, instance_norm=False, n_blocks=6):
    modules = []
    for i in range(n_blocks):
        modules += [
            ResidualBlock(
                in_channels * expansion,
                in_channels,
                stride=1,
                expansion=expansion,
                instance_norm=instance_norm,
            )
        ]
    return nn.Sequential(*modules)


def decoder(in_channels, out_channels=3, expansion=4, instance_norm=False):
    modules = [
        UpResidualBlock(
            in_channels * expansion,
            32,
            stride=2,
            expansion=expansion,
            instance_norm=instance_norm,
        ),
        UpResidualBlock(
            32 * expansion,
            16,
            stride=2,
            expansion=expansion,
            instance_norm=instance_norm,
        ),
        conv_block(16 * expansion, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm),
    ]
    return SequentialDecoder(*modules)
