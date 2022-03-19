import os
from argparse import Namespace
from os import path

import pystiche_papers.bueltemeier_mst_2021 as paper
from pystiche import image, misc, optim
from pystiche_papers import utils

from utils import get_style_images_and_guides, read_image_and_guides, get_guided_images_from_dataset, image_numbers

contents = (
        # "bueltemeier",
        # "doerksen",
        # "lohweg",    TODO: beard
        # "schaede",
    )

def unmasked_training(args, style):
    image_size = 512
    images = paper.images(args.image_source_dir)

    style_image = images[style].read(size=image_size, device=args.device)
    dataset = paper.dataset(path.join(args.dataset_dir),)
    image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    hyper_parameters = paper.hyper_parameters()

    hyper_parameters.gram_style_loss.score_weight = 1e2

    transformer = paper.training(
        image_loader,
        style_image,
        instance_norm=args.instance_norm,
        hyper_parameters=hyper_parameters,
        quiet=args.quiet,
        logger=args.logger,
    )

    model_name = f"bueltemeier_2021__{style}__intaglio"
    if args.instance_norm:
        model_name += "__instance_norm"
    utils.save_state_dict(transformer, model_name, root=args.model_dir)

    for content in contents:
        content_image, content_guides = read_image_and_guides(
            images[content], device=args.device, size=image_size
        )
        output_image, _ = paper.stylization(content_image, transformer)
        output_name = f"intaglio_{content}_{style}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)

    for image_number in image_numbers:
        content_image, content_guides = get_guided_images_from_dataset(args, image_number)
        output_image, _ = paper.stylization(content_image, transformer)
        output_name = f"intaglio_{style}_{image_number}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)


def masked_training(args, style):
    styles = (
        "DM_100_1996",
        "MAD_20_2005",
        "Specimen_0_2",
        "Specimen_0_2005",
        "UHD_20_1997",
        "GBP_5_2002",
        "UAH_1_2006",
        "LRD_50_2008",
        "MAD_2000_2002",
    )
    image_size = 512
    images = paper.images(args.image_source_dir)
    style_images_and_guides = get_style_images_and_guides(style, images,
                                                          image_size, styles,
                                                          args)
    dataset = paper.mask_dataset(path.join(args.dataset_dir), )
    image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    transformer = paper.mask_training(
        image_loader,
        style_images_and_guides,
        instance_norm=args.instance_norm,
        quiet=args.quiet,
        logger=args.logger,
    )

    model_name = f"bueltemeier_2021__mask__{style}__intaglio"
    if args.instance_norm:
        model_name += "__instance_norm"
    utils.save_state_dict(transformer, model_name, root=args.model_dir)

    for content in contents:
        content_image, content_guides = read_image_and_guides(
            images[content], device=args.device, size=image_size
        )
        output_image, _ = paper.mask_stylization(content_image, content_guides,
                                                 transformer)
        output_name = f"intaglio_mask_{content}_{style}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)

    for image_number in image_numbers:
        content_image, content_guides = get_guided_images_from_dataset(args,
                                                                       image_number)
        output_image, _ = paper.mask_stylization(content_image, content_guides,
                                                 transformer)
        output_name = f"intaglio_mask_{style}_{image_number}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)


def substyle_masked_training(args, style):
    image_size = 512
    images = paper.images(args.image_source_dir)
    style_image, style_guides = read_image_and_guides(images[style], size=image_size, device=args.device)
    del style_guides["background"]
    dataset = paper.dataset(path.join(args.dataset_dir), )
    image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    hyper_parameters = paper.hyper_parameters()

    # hyper_parameters.gram_style_loss.score_weight = 1e3
    # hyper_parameters.batch_sampler.num_iterations = 240000
    # hyper_parameters.batch_sampler.batch_size = 4

    transformer = paper.substyle_mask_training(
        image_loader,
        style_image,
        style_guides,
        instance_norm=args.instance_norm,
        quiet=args.quiet,
        logger=args.logger,
    )

    model_name = f"bueltemeier_2021__substyle_mask__{style}__intaglio"
    if args.instance_norm:
        model_name += "__instance_norm"
    utils.save_state_dict(transformer, model_name, root=args.model_dir)

    for image_number in image_numbers:
        content_image, content_guides = get_guided_images_from_dataset(args,
                                                                       image_number)
        content_regions = content_guides.keys()
        delete_region = []
        for region in content_regions:
            if region not in style_guides.keys():
                delete_region.append(region)

        for region in delete_region:
            del content_guides[region]
        transformer.enable_masked_transfer()
        output_image, _ = paper.mask_stylization(content_image, content_guides,
                                                 transformer)
        output_name = f"intaglio_mask_substyle_{style}_{image_number}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)
        # create complete image with each substyle
        transformer.disable_masked_transfer()
        for region, guide in content_guides.items():

            output_image, _ = paper.mask_stylization(content_image, {region: guide},
                                                     transformer)
            output_name = f"intaglio_mask__substyle_{region}_{style}_{image_number}"
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
    instance_norm = True
    masked = True
    substyles_only = True
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
        # dataset_path = '~/datasets/celebamask/CelebAMask-HQ/'
        dataset_path = path.join(here, "data", "images", "dataset", "CelebAMask-HQ")
        dataset_dir = (
            path.join(dataset_path, "CelebAMask-HQ-mask")
            if masked
            else path.join(
                dataset_path, "CelebA-HQ-img"
            )
        )
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
        masked=masked,
        substyles_only=substyles_only,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    styles = (
        "MAD_20_2005",
        "UHD_20_1997",
        "DM_100_1996",
        "Specimen_0_2",
    )

    for style in styles:
        for state in (True, ):
            here = path.dirname(__file__)
            args.masked = state
            # dataset_path = path.join(here, "data", "images", "dataset", "CelebAMask-HQ")
            dataset_path = '~/datasets/celebamask/CelebAMask-HQ/'
            args.dataset_dir = path.join(dataset_path, "CelebA-HQ-img")
            substyle_masked_training(args, style)
            unmasked_training(args, style)
            args.dataset_dir = path.join(dataset_path, "CelebAMask-HQ-mask")
            masked_training(args, style)


