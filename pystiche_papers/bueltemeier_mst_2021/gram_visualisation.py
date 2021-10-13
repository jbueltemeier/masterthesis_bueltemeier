from os import path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
from typing import List
from torch import nn
import torch
import pystiche
from pystiche import enc, misc, image
from pystiche.image import transforms
from pystiche.image.utils import extract_num_channels
from pystiche_papers.bueltemeier_mst_2021._data import images as _images


def apply_guide(enc: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    return (enc * guide) / torch.sum(guide)


def style_transform() -> nn.Sequential:
    transforms_: List[nn.Module] = [
        transforms.RGBToFakegrayscale(),
    ]
    return nn.Sequential(*transforms_)

def calculate_gram(repr: torch.Tensor, normalize:bool = False) -> torch.Tensor:
    return pystiche.gram_matrix(repr, normalize=normalize).squeeze()

def calculate_loss(repr1: torch.Tensor, repr2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(repr1, repr2)

device = misc.get_device(device=None)
here = "C:/Users/julia/Documents/GitHub/pystiche_papers/replication/bueltemeier_mst_2021"
image_source_dir = path.join(here, "data", "images", "source")
image_size = 512
images = _images(image_source_dir)
styles = [
    "DM_100_1996",
    "MAD_20_2005",
    "Specimen_0_2",
    "UHD_20_1997",
    "GBP_5_2002",
    "UAH_1_2006",
    "LRD_50_2008",
]
multi_layer_encoder = enc.vgg19_multi_layer_encoder(
    weights="caffe", internal_preprocessing=True, allow_inplace=True
)
gram_images = {}
for style in styles:
    style_image = images[style].read(size=image_size, device=device)
    style_guides = images[style].guides.read(interpolation_mode="nearest", size=image_size, device=device)

    if extract_num_channels(style_image) == 4:
        style_image = style_image[:, :3, :, :]

    transform = style_transform()
    style_image = transform(style_image)
    # image.show_image(style_image)

    layer = "relu5_1"
    encoder = multi_layer_encoder.extract_encoder(layer).to(device)
    enc = encoder(style_image)

    regions = list(style_guides.keys())
    gram_results = {}
    gram_results['all'] = calculate_gram(enc, normalize=True)
    for region in regions:
        guide = encoder.propagate_guide(style_guides[region])
        gram = calculate_gram(apply_guide(enc, guide))
        gram_results[region] = gram
        # plt.imshow(gram.cpu().numpy(), cmap='hot')
        # plt.show()

    # for entry1, entry2 in itertools.combinations(gram_results.items(), 2):
    #     print(entry1[0])
    #     print(entry2[0])
    #     print(calculate_loss(entry1[1], entry2[1]))
    gram_images[style] = gram_results

compare_regions = ['lips', 'skin', 'hair', 'brows', 'eye']
for styleA, styleB in itertools.combinations(gram_images.keys(),2):
    print(styleA)
    print(styleB)
    for region in compare_regions:
        print(region)
        print(calculate_loss(gram_images[styleA][region], gram_images[styleB][region]))