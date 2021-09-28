import time
from typing import Callable, Optional, Union, cast, Dict, Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche import optim, misc, loss

from pystiche_papers.utils import HyperParameters
from ._utils import optimizer as _optimizer
from ._data import style_transform as _style_transform
from ._utils import hyper_parameters as _hyper_parameters
from ._transformer import MaskMSTTransformer
from ._loss import guided_perceptual_loss

__all__ = ["default_mask_transformer_optim_loop", "training", "stylization"]


def default_mask_transformer_optim_loop(
        image_loader: DataLoader,
        transformer: MaskMSTTransformer,
        criterion: nn.Module,
        criterion_update_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor], nn.Module], None],
        optimizer: Optional[Optimizer] = None,
        quiet: bool = False,
        logger: Optional[optim.OptimLogger] = None,
        log_fn: Optional[
            Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
        ] = None,
) -> nn.Module:

    if optimizer is None:
        optimizer = _optimizer(transformer)

    if logger is None:
        logger = optim.OptimLogger()

    if log_fn is None:
        log_fn = optim.default_transformer_optim_log_fn(logger, len(image_loader))

    device = next(transformer.parameters()).device

    loading_time_start = time.time()
    for batch, input_data in enumerate(image_loader, 1):
        input_image, input_guides = input_data
        regions = list(input_guides.keys())
        input_image.to(device)
        for key in input_guides.keys():
            input_guides[key] = input_guides[key].to(device)

        transformer.set_input_guides(input_guides)
        criterion_update_fn(input_image, input_guides, criterion)
        loading_time = time.time() - loading_time_start

        def closure() -> float:
            processing_time_start = time.time()
            # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
            optimizer.zero_grad()  # type: ignore[union-attr]
            output_image = transformer(input_image, regions)
            loss = criterion(output_image)
            loss.backward()

            processing_time = time.time() - processing_time_start

            if not quiet:
                batch_size = input_image.size()[0]
                image_loading_velocity = batch_size / max(loading_time, 1e-6)
                image_processing_velocity = batch_size / max(processing_time, 1e-6)
                # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
                log_fn(batch, loss, image_loading_velocity, image_processing_velocity)  # type: ignore[misc]

            return cast(float, loss.item())

        optimizer.step(closure)
        loading_time_start = time.time()

    return transformer


def training(
        content_image_loader: DataLoader,
        style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        instance_norm: bool = False,
        hyper_parameters: Optional[HyperParameters] = None,
        quiet: bool = False,
        logger: Optional[optim.OptimLogger] = None,
        log_fn: Optional[
            Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
        ] = None,
) -> MaskMSTTransformer:

    device = misc.get_device()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    regions = list(style_images_and_guides.keys())
    transformer = MaskMSTTransformer(regions, instance_norm=instance_norm)
    transformer = transformer.train().to(device)

    criterion = guided_perceptual_loss(regions, hyper_parameters=hyper_parameters)
    criterion = criterion.eval().to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)

    for region, (image, guide) in style_images_and_guides.items():
        criterion.set_style_guide(region, guide)
        criterion.set_style_image(region, style_transform(image))

    def criterion_update_fn(input_image: torch.Tensor, input_guides: Dict[str, torch.Tensor], criterion: nn.Module) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(input_image)
        for region, guide in input_guides.items():
            cast(loss.PerceptualLoss, criterion).set_content_guide(region, guide)

    return default_mask_transformer_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def stylization(
        input_image: torch.Tensor,
        input_guides: Dict[str, torch.Tensor],
        transformer: MaskMSTTransformer,
) -> torch.Tensor:

    device = input_image.device
    transformer = transformer.eval()
    transformer = transformer.to(device)

    with torch.no_grad():
        regions = list(input_guides.keys())
        transformer.set_input_guides(input_guides)
        output_image = transformer(input_image, regions)

    return cast(torch.Tensor, output_image).detach()