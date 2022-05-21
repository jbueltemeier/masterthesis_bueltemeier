from typing import Optional, Sequence, Tuple
from torch import optim, nn

from pystiche import enc
from pystiche_papers.gatys_ecker_bethge_2016 import (
    compute_layer_weights as _compute_layer_weights,
)
from pystiche_papers.utils import HyperParameters

__all__ = [
    "multi_layer_encoder",
    "optimizer",
    "compute_layer_weights",
    "hyper_parameters",
]


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    return enc.vgg19_multi_layer_encoder(
        framework="caffe", internal_preprocessing=True, allow_inplace=True
    )


def optimizer(transformer: nn.Module) -> optim.Adam:
    return optim.Adam(transformer.parameters(), lr=1e-4)


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
        layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    return _compute_layer_weights(layers, multi_layer_encoder=multi_layer_encoder)


def hyper_parameters() -> HyperParameters:
    style_loss_layers = ("relu1_1", "relu2_1", "relu3_1")
    style_loss = HyperParameters(
        layers=style_loss_layers,
        layer_weights=compute_layer_weights(style_loss_layers),
        score_weight=1e0,
    )

    return HyperParameters(
        content_loss=HyperParameters(
            layer="relu2_2",
            score_weight=1e0,
        ),
        gram_style_loss=style_loss,
        guided_style_loss=style_loss.new_similar(
            region_weights="sum"
        ),
        mrf_style_loss=HyperParameters(
            layers=("relu3_1", "relu4_1"),
            layer_weights="mean",
            patch_size=3,
            stride=2,
            score_weight=1e-4,
        ),
        content_transform=HyperParameters(
            image_size=512,
            edge="short"
        ),
        style_transform=HyperParameters(
            image_size=512,
            edge="short"
        ),
        batch_sampler=HyperParameters(num_iterations=30000, batch_size=1),
        loss=HyperParameters(modes=["gram",]),  # possible modes "gram", "mrf", "gabor"
        masked=HyperParameters(straighten_blocks=0),
    )
