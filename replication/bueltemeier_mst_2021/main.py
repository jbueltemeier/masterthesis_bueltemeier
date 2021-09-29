import os
from argparse import Namespace
from os import path

import pystiche_papers.bueltemeier_mst_2021 as paper
from pystiche import image, misc, optim
from pystiche_papers import utils


def read_image_and_guides(image, **read_kwargs):
    return image.read(**read_kwargs), image.guides.read(**read_kwargs)


def training(args):
    image_size = 256
    contents = (
        "bueltemeier",
        "doerksen",
        # "lohweg",    TODO: beard
        "schaede", )
    styles = (
        "DM_100_1996",
        "MAD_20_2005",
        "Specimen_0_2",
        "Specimen_0_2005",
        "UHD_20_1997",
        "GBP_5_2002",
        "UAH_1_2006",
        "LRD_50_2008",
        "MAD_2000_2002")

    images = paper.images(args.image_source_dir)
    style = 'MAD_20_2005'
    style_images = {style: read_image_and_guides(images[style], device=args.device, size=image_size) for style in styles}
    style_images_and_guides = {
        'background': (style_images[style][0], style_images[style][1]["background"]),
        'skin': (style_images[style][0], style_images[style][1]["skin"]),
        'nose': (style_images[style][0], style_images[style][1]["nose"]),
        'glasses': (style_images["LRD_50_2008"][0], style_images["LRD_50_2008"][1]["accessoire"]),
        'eye': (style_images[style][0], style_images[style][1]["eye"]),
        'brows': (style_images[style][0], style_images[style][1]["brows"]),
        'ears': (style_images[style][0], style_images[style][1]["ears"]),
        'lips': (style_images[style][0], style_images[style][1]["lips"]),
        'hair': (style_images[style][0], style_images[style][1]["hair"]),
        'headwear': (style_images["MAD_2000_2002"][0], style_images["MAD_2000_2002"][1]["headwear"]),
        'accessoire': (style_images["GBP_5_2002"][0], style_images["GBP_5_2002"][1]["accessoire"]),
        'body': (style_images[style][0], style_images[style][1]["body"]),
    }

    dataset = paper.dataset(path.join(args.dataset_dir),)
    image_loader = paper.image_loader(dataset, pin_memory=str(args.device).startswith("cuda"),)

    transformer = paper.training(
        image_loader,
        style_images_and_guides,
        instance_norm=args.instance_norm,
        quiet=args.quiet,
        logger=args.logger,
    )

    model_name = f"bueltemeier_mst_2021__intaglio"
    if args.instance_norm:
        model_name += "__instance_norm"
    utils.save_state_dict(transformer, model_name, root=args.model_dir)

    # stylise some images from dataset
    iter_loader = iter(image_loader)
    for i in range(10):
        content_image, content_guides = next(iter_loader)
        output_image = paper.stylization(
            content_image,
            content_guides,
            transformer
        )
        output_name = f"intaglio_random_content_{i}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)

    for content in contents:
        content_image, content_guides = read_image_and_guides(images[content], device=args.device, size=image_size)
        output_image = paper.stylization(
            content_image,
            content_guides,
            transformer
        )
        output_name = f"intaglio_{content}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    dataset_dir = None
    model_dir = None
    device = None
    instance_norm = False
    quiet = False

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "data", "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "data", "images", "results")
    image_results_dir = process_dir(image_results_dir)

    if dataset_dir is None:
        dataset_dir = path.join(here, "data", "images", "dataset", "CelebAMask-HQ", "CelebAMask-HQ-mask")
    dataset_dir = process_dir(dataset_dir)

    if model_dir is None:
        model_dir = path.join(here, "data", "models")
    model_dir = process_dir(model_dir)

    device = misc.get_device(device=device)
    logger = optim.OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        device=device,
        instance_norm=instance_norm,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    training(args)