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
        weights="caffe", internal_preprocessing=True, allow_inplace=True
    )


def optimizer(transformer: nn.Module) -> optim.Adam:
    return optim.Adam(transformer.parameters(), lr=1e-3)


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
        layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    return _compute_layer_weights(layers, multi_layer_encoder=multi_layer_encoder)


def hyper_parameters() -> HyperParameters:
    style_loss_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    style_loss = HyperParameters(
        layers=style_loss_layers,
        layer_weights=compute_layer_weights(style_loss_layers),
        score_weight=1e3,
    )

    return HyperParameters(
        content_loss=HyperParameters(
            layer="relu4_2",
            score_weight=1e0,
        ),
        style_loss=style_loss,
        guided_style_loss=style_loss.new_similar(
            region_weights="sum"
        ),
        content_transform=HyperParameters(
            image_size=512,
            edge="short"
        )
    )
