from os import path
import os
from typing import List, Optional, Sized, Callable, Any, cast, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pystiche.data import LocalImage, LocalImageCollection
from pystiche.image import transforms, read_image, read_guides
from pystiche.image.utils import extract_num_channels
from pystiche_papers.utils import HyperParameters

from ..data.utils import FiniteCycleBatchSampler
from ._utils import hyper_parameters as _hyper_parameters

from torchvision.datasets.folder import is_image_file


__all__ = [
    "content_transform",
    "style_transform",
    "images",
    "dataset",
    "batch_sampler",
    "image_loader",
]


class OptionalRGBAToRGB(transforms.Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if extract_num_channels(x) == 4:
            return x[:, :3, :, :]
        return x


def content_transform(
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    image_size = hyper_parameters.content_transform.image_size
    transforms_: List[nn.Module] = [
        transforms.Resize(image_size, edge=hyper_parameters.content_transform.edge),
        transforms.CenterCrop((image_size, image_size)),
        # A grayscale transformer is trained. For the criterion these images have to be
        # converted into a fakegrayscale image.
        # See criterion_update_fn in _nst.py and transformer output in _modules.py.
        transforms.RGBToGrayscale(),
    ]
    return nn.Sequential(*transforms_)


def style_transform(
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transforms_: List[nn.Module] = [
        transforms.Resize(
            hyper_parameters.style_transform.edge_size,
            edge=hyper_parameters.style_transform.edge,
        ),
        OptionalRGBAToRGB(),
        transforms.RGBToGrayscale(),
    ]
    return nn.Sequential(*transforms_)


def images(root: str) -> LocalImageCollection:

    content_root = path.join(root, "content/")
    style_root = path.join(root, "style/")
    content_images = {
        "karya": LocalImage(path.join(content_root, "karya.jpg"),),
        "004": LocalImage(path.join(content_root, "004.jpg"),),
        "04": LocalImage(path.join(content_root, "04.jpg"),),
        "bueltemeier": LocalImage(path.join(content_root, "bueltemeier.png"),),
    }

    style_images = {
        "DM100": LocalImage(path.join(style_root, "DM_100_1996.png"),),
        "MAD20": LocalImage(path.join(style_root, "MAD_20_2005.png"),),
        "Specimen0": LocalImage(path.join(style_root, "Specimen_0_2.png"),),
        "UHD20": LocalImage(path.join(style_root, "UHD_20_1997.png"),),
    }
    return LocalImageCollection({**content_images, **style_images},)


class GuidesImageFolderDataset(Dataset):
    def __init__(
            self,
            root: str,
            transform: Optional[nn.Module] = None,
            mask_transform: Optional[nn.Module] = None,
            importer: Optional[Callable[[str], Any]] = None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.folders = self._collect_folder_paths()
        self.transform = transform
        self.mask_transform = mask_transform

        if importer is None:
            def importer(folder: Tuple[str, str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                return read_image(folder[0], make_batched=False), read_guides(folder[1], make_batched=False)

        self.importer = cast(Callable[[Tuple[str, str]], Any], importer)

    def _collect_folder_paths(self) -> List[Tuple[str, str]]:
        folders = [
                (path.join(self.root, folder, subfile), path.join(self.root, folder, 'guides'))
                for folder in os.listdir(self.root)
                for subfile in os.listdir(path.join(self.root, folder)) if is_image_file( subfile)
                if 'guides' in os.listdir(path.join(self.root, folder)) and len(os.listdir(path.join(self.root, folder))) == 2
            ]

        if len(folders) == 0:
            msg = f"The directory {self.root} does not contain any folders with a " \
                  f"image and a guides folder."
            raise RuntimeError(msg)

        return folders

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int) -> Any:
        folder = self.folders[idx]
        image, guides = self.importer(folder)

        if self.transform:
            image = self.transform(image)
            guides = [self.mask_transform(guide) for guide in guides]

        return image, guides


def dataset(root: str, transform: Optional[nn.Module] = None,) -> GuidesImageFolderDataset:
    if transform is None:
        transform = content_transform()

    return GuidesImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized, hyper_parameters: Optional[HyperParameters] = None,
) -> FiniteCycleBatchSampler:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return FiniteCycleBatchSampler(
        data_source,
        num_batches=hyper_parameters.batch_sampler.num_batches,
        batch_size=hyper_parameters.batch_sampler.batch_size,
    )


def image_loader(dataset: Dataset, pin_memory: bool = True,) -> DataLoader:
    return DataLoader(
        dataset,
        # batch_sampler=batch_sampler(dataset),
        num_workers=0,
        pin_memory=pin_memory,
    )
