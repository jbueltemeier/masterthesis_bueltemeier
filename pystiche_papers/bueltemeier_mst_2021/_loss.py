from copy import copy
from typing import Optional, Sequence, List, Callable, Union

import torch
import pystiche
from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "gram_style_loss",
    "guided_style_loss",
    "perceptual_loss",
    "FlexibleGuidedPerceptualLoss",
    "guided_perceptual_loss",
]


def content_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.FeatureReconstructionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return ops.FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


def mrf_style_loss(
        multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
        hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.MRFOperator:
        return ops.MRFOperator(
            encoder,
            hyper_parameters.mrf_style_loss.patch_size,  # type: ignore[union-attr]
            stride=hyper_parameters.mrf_style_loss.stride,  # type: ignore[union-attr]
            score_weight=layer_weight,
        )

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.mrf_style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.mrf_style_loss.layer_weights,
        score_weight=hyper_parameters.mrf_style_loss.score_weight,
    )


def gram_style_loss(
        multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
        hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight)

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.gram_style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.gram_style_loss.layer_weights,
        score_weight=hyper_parameters.gram_style_loss.score_weight,
    )


class MultiRegionOperator(ops.MultiRegionOperator):
    def __init__(
            self,
            regions: Sequence[str],
            get_op: Callable[[str, float], ops.Operator],
            region_weights: Union[str, Sequence[float]] = "sum",
            score_weight: float = 1e0,
    ):
        super().__init__(
            regions, get_op, region_weights=region_weights, score_weight=score_weight,
        )
        self.input_regions: List[str] = []

    def set_input_regions(self, regions: torch.Tensor) -> None:
        self.input_regions = regions

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children() if name in self.input_regions]
        )


def guided_style_loss(
    regions: Sequence[str],
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiRegionOperator:

    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    # the copy is needed here in order to not override style_loss.score_weight later
    hyper_parameters = (
        _hyper_parameters() if hyper_parameters is None else copy(hyper_parameters)
    )

    def get_region_op(
        region: str, region_weight: float
    ) -> ops.MultiLayerEncodingOperator:
        hyper_parameters.style_loss.score_weight = region_weight  # type: ignore[union-attr]
        return gram_style_loss(
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters.new_similar(),  # type: ignore[union-attr]
        )

    return MultiRegionOperator(
        regions,
        get_region_op,
        region_weights=hyper_parameters.guided_style_loss.region_weights,
        score_weight=hyper_parameters.guided_style_loss.score_weight,
    )


def perceptual_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()


    style_losses = []
    if "gram" in hyper_parameters.loss.modes:
        style_losses.append(
            (
                "gram_loss",
                gram_style_loss(
                    multi_layer_encoder=multi_layer_encoder,
                    hyper_parameters=hyper_parameters,
                ),
            )
        )

    if "mrf" in hyper_parameters.loss.modes:
        style_losses.append(
            (
                "mrf_loss",
                mrf_style_loss(
                    multi_layer_encoder=multi_layer_encoder,
                    hyper_parameters=hyper_parameters,
                ),
            )
        )


    style_loss = (
        ops.OperatorContainer(style_losses)
        if len(style_losses) != 1
        else style_losses[0][-1]
    )

    return loss.PerceptualLoss(
        content_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
        style_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
    )


class FlexibleGuidedPerceptualLoss(loss.GuidedPerceptualLoss):
    def set_input_regions(self, regions: List[str]) -> None:
        r"""Set the current

        Args:
            regions: List of the regions.
        """
        self.style_loss.input_regions = regions


def guided_perceptual_loss(
    regions: Sequence[str],
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> FlexibleGuidedPerceptualLoss:

    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return FlexibleGuidedPerceptualLoss(
        content_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
        guided_style_loss(
            regions,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
    )
