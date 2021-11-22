from os import path
import os
from typing import List, Optional, Sized, Callable, Any, cast, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pystiche import data
from pystiche.image import transforms, read_image, read_guides
from pystiche.image.utils import extract_num_channels
from pystiche_papers.utils import HyperParameters

from ..data.utils import RandomNumIterationsBatchSampler
from ._utils import hyper_parameters as _hyper_parameters

from torchvision.datasets.folder import is_image_file


__all__ = [
    "content_transform",
    "content_mask_transform",
    "style_transform",
    "style_mask_transform",
    "images",
    "nst_images",
    "dataset",
    "mask_dataset",
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


def content_mask_transform(
        hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    image_size = hyper_parameters.content_transform.image_size
    transforms_: List[nn.Module] = [
        transforms.Resize(image_size, edge=hyper_parameters.content_transform.edge, interpolation_mode="nearest"),
        transforms.CenterCrop((image_size, image_size)),
    ]
    return nn.Sequential(*transforms_)


def style_transform(
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    image_size = hyper_parameters.style_transform.image_size
    transforms_: List[nn.Module] = [
        transforms.Resize(
            image_size,
            edge=hyper_parameters.style_transform.edge,
        ),
        transforms.CenterCrop((image_size, image_size)),
        OptionalRGBAToRGB(),
        transforms.RGBToGrayscale(),
    ]
    return nn.Sequential(*transforms_)


def style_mask_transform(
        hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    image_size = hyper_parameters.style_transform.image_size
    transforms_: List[nn.Module] = [
        transforms.Resize(
            image_size,
            edge=hyper_parameters.style_transform.edge,
            interpolation_mode="nearest"
        ),
        transforms.CenterCrop((image_size, image_size)),
    ]
    return nn.Sequential(*transforms_)


def images(root: str) -> data.LocalImageCollection:

    guide_images = {
        "bueltemeier": data.LocalImage(
            file=path.join(root, "bueltemeier.png"),
            collect_local_guides=True
        ),
        "doerksen": data.LocalImage(
            file=path.join(root, "doerksen.jpg"),
            collect_local_guides=True
        ),
        "lohweg": data.LocalImage(
            file=path.join(root, "Lohweg_glasses.png"),
            collect_local_guides=True
        ),
        "schaede": data.LocalImage(
            file=path.join(root, "Schaede_glasses.jpg"),
            collect_local_guides=True
        ),
        "DM_100_1996": data.LocalImage(
            file=path.join(root, "DM_100_1996.png"),
            collect_local_guides=True
        ),
        "MAD_20_2005": data.LocalImage(
            file=path.join(root, "MAD_20_2005.png"),
            collect_local_guides=True
        ),
        "Specimen_0_2": data.LocalImage(
            file=path.join(root, "Specimen_0_2.png"),
            collect_local_guides=True
        ),
        "Specimen_0_2005": data.LocalImage(
            file=path.join(root, "Specimen_0_2005.png"),
            collect_local_guides=True
        ),
        "UHD_20_1997": data.LocalImage(
            file=path.join(root, "UHD_20_1997.png"),
            collect_local_guides=True
        ),
        "GBP_5_2002": data.LocalImage(
            file=path.join(root, "GBP_5_2002.png"),
            collect_local_guides=True
        ),
        "UAH_1_2006": data.LocalImage(
            file=path.join(root, "UAH_1_2006.png"),
            collect_local_guides=True
        ),
        "LRD_50_2008": data.LocalImage(
            file=path.join(root, "LRD_50_2008.png"),
            collect_local_guides=True
        ),
        "MAD_2000_2002": data.LocalImage(
            file=path.join(root, "MAD_2000_2002.png"),
            collect_local_guides=True
        ),
    }
    return data.LocalImageCollection({**guide_images},)


def nst_images():
    images_ = {
        "sky": data.DownloadableImage(
            "https://free-images.com/md/2397/abstract_astrology_astronomy_315181.jpg",
            license=data.PublicDomainLicense(),
            md5="87ac575a293d7d669874b43070675b2d",
            file="abstract_astrology_astronomy_315181.jpg",
        ),
        "mosaic": data.DownloadableImage(
            "https://free-images.com/md/1278/mosaic_ducks_massimo.jpg",
            license=data.PublicDomainLicense(),
            md5="4d429a89f9d7a62f94e87e1f607f221c",
            file="mosaic_ducks_massimo.jpg",
        ),
        "mosaic2": data.DownloadableImage(
            "https://free-images.com/md/ff61/mosaic_color_colorful_pattern.jpg",
            license=data.PublicDomainLicense(),
            md5="ce766692a6048d6c9ea8385bc08cd543",
            file="mosaic_color_colorful_pattern.jpg",
        ),
        "abstract": data.DownloadableImage(
            "https://free-images.com/md/5b7f/abstract_art_colorful_1085200.jpg",
            license=data.PublicDomainLicense(),
            md5="efa658b17912e01a9685c68c4fc64aec",
            file="abstract_art_colorful_1085200.jpg",
        ),
        "abstract2": data.DownloadableImage(
            "https://free-images.com/md/1f5a/art_color_abstract_design.jpg",
            license=data.PublicDomainLicense(),
            md5="63dce4b139cd71e119a938a7ba6f68d7",
            file="art_color_abstract_design.jpg",
        ),
        "bird": data.DownloadableImage(
            "https://free-images.com/lg/2b9d/bird_wildlife_sky_clouds.jpg",
            license=data.PublicDomainLicense(),
            md5="79da0d0b491678e34360973335147a17",
            file="bird_wildlife_sky_clouds.jpg",
        ),
        "flower": data.DownloadableImage(
            "https://free-images.com/lg/19a7/little_orchid_d1212_light.jpg",
            license=data.PublicDomainLicense(),
            md5="d252e56077a239c7eed18e31cbff0ac3",
            file="little_orchid_d1212_light.jpg",
        ),
        "mountain": data.DownloadableImage(
            "https://free-images.com/md/ae78/landscape_nature_sky_clouds.jpg",
            license=data.PublicDomainLicense(),
            md5="ac7e753b0226929a07afc674abeb8a3b",
            file="landscape_nature_sky_clouds.jpg",
        ),
        "shipwreck": data.DownloadableImage(
            "https://blog-imgs-51.fc2.com/b/e/l/bell1976brain/800px-Shipwreck_turner.jpg",
            title="Shipwreck of the Minotaur",
            author="J. M. W. Turner",
            date="ca. 1810",
            license=data.ExpiredCopyrightLicense(1851),
            md5="4fb76d6f6fc1678cb74e858324d4d0cb",
            file="shipwreck_of_the_minotaur__turner.jpg",
        ),
        "starry_night": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            title="Starry Night",
            author="Vincent van Gogh",
            date="ca. 1889",
            license=data.ExpiredCopyrightLicense(1890),
            md5="372e5bc438e3e8d0eb52cc6f7ef44760",
        ),
        "the_scream": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
            title="The Scream",
            author="Edvard Munch",
            date="ca. 1893",
            license=data.ExpiredCopyrightLicense(1944),
            md5="46ef64eea5a7b2d13dbadd420b531249",
        ),
        "femme_nue_assise": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg",
            title="Figure dans un Fauteuil",
            author="Pablo Ruiz Picasso",
            date="ca. 1909",
            license=data.ExpiredCopyrightLicense(1973),
            md5="ba14b947b225d9e5c59520a814376944",
        ),
        "composition_vii": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
            title="Composition VII",
            author="Wassily Kandinsky",
            date="1913",
            license=data.ExpiredCopyrightLicense(1944),
            md5="bfcbc420684bf27d2d8581fa8cc9522f",
        ),
    }
    return data.DownloadableImageCollection(images_)


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
            for guide_name, guide in guides.items():
                guides[guide_name] = self.mask_transform(guide)

        return image, guides


def dataset(root: str, transform: Optional[nn.Module] = None) -> data.ImageFolderDataset:
    if transform is None:
        transform = content_transform()

    return data.ImageFolderDataset(root, transform=transform)


def mask_dataset(root: str, transform: Optional[nn.Module] = None, mask_transform: Optional[nn.Module] = None,) -> GuidesImageFolderDataset:
    if transform is None:
        transform = content_transform()

    if mask_transform is None:
        mask_transform = content_mask_transform()

    return GuidesImageFolderDataset(root, transform=transform, mask_transform=mask_transform)


def batch_sampler(
    data_source: Sized, hyper_parameters: Optional[HyperParameters] = None,
) -> RandomNumIterationsBatchSampler:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return RandomNumIterationsBatchSampler(
        data_source,
        num_iterations=hyper_parameters.batch_sampler.num_iterations,
        batch_size=hyper_parameters.batch_sampler.batch_size,
    )


def image_loader(dataset: Dataset, pin_memory: bool = True,) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler(dataset),
        num_workers=0,
        pin_memory=pin_memory,
    )
