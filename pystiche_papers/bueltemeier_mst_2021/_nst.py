import time
from typing import Callable, Optional, Union, cast, Dict, Tuple, List, Iterable, Iterator
# from memory_profiler import profile

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import nn
import pystiche
from pystiche import optim, misc, loss, image, ops
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale

from pystiche_papers.utils import HyperParameters
from ._utils import optimizer as _optimizer
from ._data import style_transform as _style_transform
from ._data import style_mask_transform as _style_mask_transform
from ._data import content_transform as _content_transform
from ._data import content_mask_transform as _content_mask_transform
from ._utils import hyper_parameters as _hyper_parameters
from ._transformer import MSTTransformer, MaskMSTTransformer, SubstyleMSTTransformer
from ._loss import perceptual_loss, guided_perceptual_loss, FlexibleGuidedPerceptualLoss

__all__ = [
    "default_mask_transformer_optim_loop",
    "training",
    "substyle_mask_training",
    "mask_training",
    "stylization",
    "mask_stylization"]

def unsupervise(image_loader: Iterable) -> Iterator[torch.Tensor]:
    for input in image_loader:
        if isinstance(input, (tuple, list)):
            input = input[0]

        yield input

def default_mask_substyle_transformer_optim_loop(
    image_loader: DataLoader,
    transformer: SubstyleMSTTransformer,
    regions: List[str],
    criterion: FlexibleGuidedPerceptualLoss,
    criterion_update_fn: Optional[Callable[[torch.Tensor, FlexibleGuidedPerceptualLoss, List[str]], None]],
    optimizer: Optional[Optimizer] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> SubstyleMSTTransformer:
    if optimizer is None:
        optimizer = _optimizer(transformer)

    if logger is None:
        logger = optim.OptimLogger()

    if log_fn is None:
        log_fn = optim.default_transformer_optim_log_fn(logger, len(image_loader))
    regions = transformer.regions

    device = next(transformer.parameters()).device

    loading_time_start = time.time()
    transformer.disable_masked_transfer()
    for batch, input_image in enumerate(unsupervise(image_loader), 1):
        input_image = input_image.to(device)
        region = [regions[batch % len(regions)]]
        criterion_update_fn(input_image, criterion, region)
        loading_time = time.time() - loading_time_start

        def closure() -> float:
            processing_time_start = time.time()
            # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
            optimizer.zero_grad()  # type: ignore[union-attr]
            output_image = transformer(input_image, region)
            loss = criterion(grayscale_to_fakegrayscale(output_image))
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


def default_mask_transformer_optim_loop(
    image_loader: DataLoader,
    transformer: MaskMSTTransformer,
    criterion: FlexibleGuidedPerceptualLoss,
    criterion_update_fn: Callable[
        [torch.Tensor, Dict[str, torch.Tensor], FlexibleGuidedPerceptualLoss], None
    ],
    optimizer: Optional[Optimizer] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> MaskMSTTransformer:

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
        input_image = input_image.to(device)
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
            loss = criterion(grayscale_to_fakegrayscale(output_image))
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
    style_image: torch.Tensor,
    instance_norm: bool = False,
    hyper_parameters: Optional[HyperParameters] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:

    device = misc.get_device()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transformer = MSTTransformer(in_channels=1, instance_norm=instance_norm)
    transformer = transformer.train().to(device)

    criterion = perceptual_loss(hyper_parameters=hyper_parameters)
    criterion = criterion.eval().to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    transformer.set_target_image(style_image)

    if isinstance(type(criterion.style_loss), ops.OperatorContainer.__class__):
        for op in criterion.style_loss.operators():
            op.set_target_image(grayscale_to_fakegrayscale(style_image))
    else:
        criterion.set_style_image(grayscale_to_fakegrayscale(style_image))

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(
            grayscale_to_fakegrayscale(input_image)
        )

    return optim.default_transformer_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def substyle_mask_training(
    content_image_loader: DataLoader,
    style_image: torch.Tensor,
    style_guides: Dict[str,torch.Tensor],
    instance_norm: bool = False,
    hyper_parameters: Optional[HyperParameters] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> SubstyleMSTTransformer:

    device = misc.get_device()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    regions = list(style_guides.keys())
    transformer = SubstyleMSTTransformer(
        regions, instance_norm=instance_norm, in_channels=1
    )
    transformer = transformer.train().to(device)

    criterion = guided_perceptual_loss(regions, hyper_parameters=hyper_parameters)
    criterion = criterion.eval().to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)

    style_image = style_transform(style_image)

    style_mask_transform = _style_mask_transform(hyper_parameters=hyper_parameters)
    style_mask_transform = style_mask_transform.to(device)

    for region, guide in style_guides.items():
        transformer.set_target_image(style_image, region)
        criterion.set_style_image(region, grayscale_to_fakegrayscale(style_image))
        guide = style_mask_transform(guide)
        transformer.set_target_guide(guide, region)
        criterion.set_style_guide(region, guide)

    def criterion_update_fn(
        input_image: torch.Tensor,
        criterion: FlexibleGuidedPerceptualLoss,
        region: List[str]
    ) -> None:
        cast(FlexibleGuidedPerceptualLoss, criterion).set_content_image(
            grayscale_to_fakegrayscale(input_image)
        )
        image_size = image.extract_image_size(input_image)
        cast(FlexibleGuidedPerceptualLoss, criterion).set_content_guide(
            region[0], torch.ones([1, 1, image_size[0],image_size[1]]).to(input_image.device)
        )
        cast(FlexibleGuidedPerceptualLoss, criterion).set_input_regions(
            region
        )

    return default_mask_substyle_transformer_optim_loop(
        content_image_loader,
        transformer,
        regions,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def mask_training(
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
    transformer = MaskMSTTransformer(
        regions, instance_norm=instance_norm, in_channels=1
    )
    transformer = transformer.train().to(device)

    criterion = guided_perceptual_loss(regions, hyper_parameters=hyper_parameters)
    criterion = criterion.eval().to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)

    style_mask_transform = _style_mask_transform(hyper_parameters=hyper_parameters)
    style_mask_transform = style_mask_transform.to(device)

    for region, (image, guide) in style_images_and_guides.items():
        image = style_transform(image)
        transformer.set_target_image(image, region)
        criterion.set_style_image(region, grayscale_to_fakegrayscale(image))
        guide = style_mask_transform(guide)
        transformer.set_target_guide(guide, region)
        criterion.set_style_guide(region, guide)

    def criterion_update_fn(
        input_image: torch.Tensor,
        input_guides: Dict[str, torch.Tensor],
        criterion: FlexibleGuidedPerceptualLoss,
    ) -> None:
        cast(FlexibleGuidedPerceptualLoss, criterion).set_content_image(
            grayscale_to_fakegrayscale(input_image)
        )
        for region, guide in input_guides.items():
            cast(FlexibleGuidedPerceptualLoss, criterion).set_content_guide(
                region, guide
            )
        cast(FlexibleGuidedPerceptualLoss, criterion).set_input_regions(
            list(input_guides.keys())
        )

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


# @profile
def stylization(
    input_image: torch.Tensor,
    transformer: nn.Module,
    hyper_parameters: Optional[HyperParameters] = None,
    track_time:bool = False
) -> Tuple[torch.Tensor, Optional[int]]:

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    image_size = hyper_parameters.content_transform.image_size
    device = input_image.device
    transformer = transformer.eval()
    transformer = transformer.to(device)

    if track_time:
        start_epoch = time.time()
    else:
        execution_time = None
    if image.extract_num_channels(input_image) == 3 or image.extract_image_size(
        input_image
    ) != (image_size, image_size):
        content_transform = _content_transform(hyper_parameters=hyper_parameters)
        content_transform = content_transform.to(device)

        input_image = content_transform(input_image)

    if len(input_image.size()) == 3:
        input_image = input_image.unsqueeze(dim=0)

    with torch.no_grad():
        output_image = transformer(input_image)
        if track_time:
            torch.cuda.synchronize()
            end_epoch = time.time()
            execution_time = end_epoch - start_epoch

    return cast(torch.Tensor, output_image).detach(), execution_time



# @profile
def mask_stylization(
    input_image: torch.Tensor,
    input_guides: Dict[str, torch.Tensor],
    transformer: MaskMSTTransformer,
    hyper_parameters: Optional[HyperParameters] = None,
    track_time:bool = False
) -> Tuple[torch.Tensor, Optional[int]]:

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    image_size = hyper_parameters.content_transform.image_size
    device = input_image.device
    transformer = transformer.eval()
    transformer = transformer.to(device)

    if track_time:
        start_epoch = time.time()
    else:
        execution_time = None

    if image.extract_num_channels(input_image) == 3 or image.extract_image_size(
        input_image
    ) != (image_size, image_size):
        content_transform = _content_transform(hyper_parameters=hyper_parameters)
        content_transform = content_transform.to(device)

        content_mask_transform = _content_mask_transform(
            hyper_parameters=hyper_parameters
        )
        content_mask_transform = content_mask_transform.to(device)
        input_image = content_transform(input_image)
        for name, guide in input_guides.items():
            input_guides[name] = content_mask_transform(guide)

    if len(input_image.size()) == 3:
        input_image = input_image.unsqueeze(dim=0)
        for key, guide in input_guides.items():
            input_guides[key] = input_guides[key].unsqueeze(dim=0)

    with torch.no_grad():
        regions = list(input_guides.keys())
        transformer.set_input_guides(input_guides)
        output_image = transformer(input_image, regions)

        if track_time:
            torch.cuda.synchronize()
            end_epoch = time.time()
            execution_time = end_epoch - start_epoch

    return cast(torch.Tensor, output_image).detach(), execution_time
