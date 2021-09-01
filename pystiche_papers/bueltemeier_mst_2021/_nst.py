import time
import warnings
from typing import Callable, Optional, Union, cast, Dict, Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche.misc import build_deprecation_message
from pystiche import optim, misc, loss

from ._data import images as _images
from pystiche_papers.utils import HyperParameters
from ._utils import optimizer as _optimizer
from ._data import style_transform as _style_transform
from ._utils import hyper_parameters as _hyper_parameters
from ._transformer import MSTTransformer as _transformer
from ._loss import perceptual_loss, guided_perceptual_loss


def default_mask_transformer_optim_loop(
        image_loader: DataLoader,
        transformer: nn.Module,
        criterion: nn.Module,
        criterion_update_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor], nn.Module], None],
        optimizer: Optional[Optimizer] = None,
        get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
        quiet: bool = False,
        logger: Optional[optim.OptimLogger] = None,
        log_fn: Optional[
            Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
        ] = None,
) -> nn.Module:
    if isinstance(transformer, torch.device):  # type: ignore[unreachable]
        msg = (  # type: ignore[unreachable]
            "The parameter device was removed in 0.4.0. It is now extracted out of "
            "the transformer parameters."
        )
        raise RuntimeError(msg)

    if get_optimizer is not None:
        msg = build_deprecation_message(
            "The parameter get_optimizer",
            "0.4.0",
            info="You can achieve the same functionality by passing optimizer=get_optimizer(transformer).",
            url="https://github.com/pmeier/pystiche/pull/96",
        )
        warnings.warn(msg)
        optimizer = get_optimizer(transformer)

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
        input_image.to(device)
        for key in input_guides.keys():
            input_guides[key] = input_guides[key].to(device)

        criterion_update_fn(input_image, input_guides, criterion)

        loading_time = time.time() - loading_time_start

        def closure() -> float:
            processing_time_start = time.time()

            # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
            optimizer.zero_grad()  # type: ignore[union-attr]

            # TODO: set guides in Transformer
            output_image = transformer(input_image)
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
        instance_norm: Optional[bool] = None,
        hyper_parameters: Optional[HyperParameters] = None,
        quiet: bool = False,
        logger: Optional[optim.OptimLogger] = None,
        log_fn: Optional[
            Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
        ] = None,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image: Style image on which the ``transformer`` should be trained. If
            ``str``, the image is read from
            :func:`~pystiche_papers.johnson_alahi_li_2016.images`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. If omitted,
            defaults to ``impl_params``.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If omitted,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If omitted,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    """

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transformer = _transformer(instance_norm=instance_norm)
    transformer = transformer.train().to(device)

    criterion = perceptual_loss(hyper_parameters=hyper_parameters)
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