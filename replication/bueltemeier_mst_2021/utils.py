import os
from os import path

from torch import nn
from pystiche.image import transforms

from typing import Dict
from torchvision.datasets.folder import is_image_file

from pystiche import data, image


seg_color_map = {
    'background': (153,153,153),
    'skin': (152,78,163),
    'nose': (77,175,74),
    'glasses': (255,255,255),
    'eye': (55,126,184),
    'brows': (247,129,191),
    'ears': (166,86,40),
    'lips': (255,255,51),
    'hair': (255,127,0),
    'headwear': (166,206,227),
    'accessoire': (0,0,0),
    'body': (228,26,28),
}

DM_100_1996 = {
    "background": (1,1,617,512),
    "skin": (1,1,617,512),
    "nose": (1,1,617,512),
    "glasses": (1,1,684,512),
    "eye": (1,1,617,512),
    "brows": (1,1,617,512),
    "ears": (1,1,617,512),
    "lips": (1,1,617,512),
    "hair": (1,1,617,512),
    "headwear": (1,1,687,512),
    "accessoire": (1,1,729,512),
    "body": (1,1,741,512),
}
UHD_20_1997 = {
    "background": (1,1,787,512),
    "skin": (1,1,787,512),
    "nose": (1,1,787,512),
    "glasses": (1,1,684,512),
    "eye": (1,1,787,512),
    "brows": (1,1,787,512),
    "ears": (1,1,787,512),
    "lips": (1,1,787,512),
    "hair": (1,1,787,512),
    "headwear": (1,1,687,512),
    "accessoire": (1,1,729,512),
    "body": (1,1,741,512),
}

MAD_20_1997 = {
    "background": (1,1,741,512),
    "skin": (1,1,741,512),
    "nose": (1,1,741,512),
    "glasses": (1,1,684,512),
    "eye": (1,1,741,512),
    "brows": (1,1,741,512),
    "ears": (1,1,741,512),
    "lips": (1,1,741,512),
    "hair": (1,1,741,512),
    "headwear": (1,1,687,512),
    "accessoire": (1,1,729,512),
    "body": (1,1,741,512),
}


def init_dataset(masked=True, programming_dataset=True):
    if programming_dataset:
        here = path.dirname(__file__)
        dataset_path = path.join(here, "data", "images", "dataset", "CelebAMask-HQ")
    else:  # extern --  example Rechenknecht
        dataset_path = '~/datasets/celebamask/CelebAMask-HQ/'
    return path.join(dataset_path, "CelebAMask-HQ-mask") if masked else \
        path.join(dataset_path, "CelebA-HQ-img")


def content_transform(image_size) -> nn.Sequential:
    transforms_ = [
        transforms.Resize(image_size, edge="short"),
        transforms.CenterCrop((image_size, image_size)),
    ]
    return nn.Sequential(*transforms_)


def content_mask_transform(image_size) -> nn.Sequential:
    transforms_ = [
        transforms.Resize(image_size, edge="short", interpolation_mode="nearest"),
        transforms.CenterCrop((image_size, image_size)),
    ]
    return nn.Sequential(*transforms_)


def read_image_and_guides(image, **read_kwargs):
    return (
        image.read(**read_kwargs),
        image.guides.read(interpolation_mode="nearest", **read_kwargs),
    )


def get_style_images_and_guides(style, images, image_size, styles, args):
    style_images = {
        style: read_image_and_guides(images[style], size=image_size)
        for style in styles
    }
    return {
        "background": (
            style_images[style][0].to(args.device),
            style_images[style][1]["background"].to(args.device)
        ),
        "skin": (
            style_images[style][0].to(args.device),
            style_images[style][1]["skin"].to(args.device)
        ),
        "nose": (
            style_images[style][0].to(args.device),
            style_images[style][1]["nose"].to(args.device)
        ),
        "glasses": (
            style_images["LRD_50_2008"][0].to(args.device),
            style_images["LRD_50_2008"][1]["accessoire"].to(args.device),
        ),
        "eye": (
            style_images[style][0].to(args.device),
            style_images[style][1]["eye"].to(args.device)
        ),
        "brows": (
            style_images[style][0].to(args.device),
            style_images[style][1]["brows"].to(args.device)
        ),
        "ears": (
            style_images[style][0].to(args.device),
            style_images[style][1]["ears"].to(args.device))
        if style not in ["Specimen_0_2", "Specimen_0_2005"] else (
            style_images["MAD_20_2005"][0].to(args.device),
            style_images["MAD_20_2005"][1]["eye"].to(args.device)
        ),
        "lips": (
            style_images[style][0].to(args.device),
            style_images[style][1]["lips"].to(args.device)
        ),
        "hair": (
            style_images[style][0].to(args.device),
            style_images[style][1]["hair"].to(args.device)
        ),
        "headwear": (
            style_images["MAD_2000_2002"][0].to(args.device),
            style_images["MAD_2000_2002"][1]["headwear"].to(args.device),
        ),
        "accessoire": (
            style_images["GBP_5_2002"][0].to(args.device),
            style_images["GBP_5_2002"][1]["accessoire"].to(args.device),
        ),
        "body": (
            style_images["MAD_20_2005"][0].to(args.device),
            style_images["MAD_20_2005"][1]["body"].to(args.device)
        ),
    }

image_numbers = [
    22555,23597,23620,23701,24130,24294,24409,24405,24525,24539,24602,
    # 92,249,265,356,16569,16584,17858,17931,18389,18505,18565,18568,
    # 18591,18789,19758,19912,21001,21153,21922,22287,22294,
    # 22858,22947,23269,23350,23451,23486,24142,24915,
    # 10,18,35,143,203,196,194,503,724,756,838,898,962,1222,1439,2025,
    2263,2638,2910,2094,3423,4069,4656,4922,4934,5069,5064,5087,5393,5495,
    5723,5987,6235,6591,6582,6635,6724,7344,7463,7607,7812,8062,8067,8068
]

detail_image_numbers = [
    (22555, (50,150,200,300)),
    (22555, (200,300,200,300)),
    # (22555, (670,750,120,200)),
    (23620, (20,120,220,320)),
    (23620, (200,300,200,300)),
    # (23620, (670,770,550,650)),
    # (22294, (660,760,540,640)),
    (22294, (10,110,280,380)),
    (22294, (180,280,220,320)),
]


def collect_guides(dir: str):
    image_files = [file for file in os.listdir(dir) if is_image_file(file)]
    if not image_files:
        return None

    guides: Dict[str, "data.LocalImage"] = {}
    for file in image_files:
        region = path.splitext(path.basename(file))[0]
        guides[region] = data.LocalImage(
            path.join(dir, file), collect_local_guides=False
        )
    return data.LocalImageCollection(guides)


def get_guided_images_from_dataset(args, image_number):
    root = init_dataset(masked=True,programming_dataset=args.programming_dataset)
    local_path = path.join(root,  str(image_number).rjust(5, '0'))
    images = data.LocalImageCollection(
         {
            "Image": data.LocalImage(
                file=path.join(local_path, f"{image_number}.jpg"),
            guides=collect_guides(path.join(root,  str(image_number).rjust(5, '0'), "guides"))),
            }
    )
    complete_image = images["Image"].read(size=512, device=args.device)
    guides = images["Image"].guides.read(size=512, device=args.device)
    return complete_image, guides


def crop_image_detail(image, positons):
    return image[:, :, positons[0]:positons[1], positons[2]:positons[3]]


def crop_guides_detail(guides, positons):
    reduced_guides = {}
    for name, guide in guides.items():
        guide = guide[:, :, positons[0]:positons[1], positons[2]:positons[3]]
        if not guide.sum() == 0:
            reduced_guides[name] = guide
    return reduced_guides


def crop_detail(args, umage_number=22294):
    content_image, content_guides = get_guided_images_from_dataset(args, 22294)
    content_image = crop_image_detail(content_image, (180,280,220,320))
    content_guides = crop_guides_detail(content_guides, (180,280,220,320))
    image.show_image(content_image)
