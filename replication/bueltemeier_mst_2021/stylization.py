import os
from argparse import Namespace
from os import path

import torch
from pystiche import misc, optim, image
import pystiche_papers.bueltemeier_mst_2021 as paper
from pystiche.image import write_image

def main(args):
    if args.masked:
        model_name = "bueltemeier_2021__mask__MAD_20_2005__intaglio__instance_norm-c1018c1b.pth"
        transformer = load_transformer(args.model_dir, args.instance_norm, model_name)

        dataset = paper.mask_dataset(path.join(args.dataset_dir))

        for i in range(2):
            content_image, content_guides = dataset.__getitem__(i)
            output_image = paper.mask_stylization(content_image, content_guides, transformer)

            style = model_name.split("__")[1]
            output_name = f"intaglio_mask_{i}_{style}__{args.device}"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)
    else:
        model_name = "bueltemeier_2021__MAD_20_2005__intaglio__instance_norm-1856e0fa.pth"
        transformer = load_transformer(args.model_dir, args.instance_norm, model_name)

        dataset = paper.dataset(path.join(args.dataset_dir))

        for i in range(2):
            content_image = dataset.__getitem__(i)
            output_image = paper.stylization(content_image, transformer)

            style = model_name.split("__")[1]
            output_name = f"intaglio_{i}_{style}__{args.device}"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)



def load_transformer(model_dir, instance_norm, model_name):
    def load(regions=None):
        if len(regions) > 1:
            return paper.MaskMSTTransformer(regions, in_channels=1, instance_norm=instance_norm, recalc_enc=False)
        else:
            return paper.MSTTransformer(in_channels=1, instance_norm=instance_norm)

    state_dict = load_local_weights(model_dir, model_name)

    # remove the saved conditions from input
    remove_keys = []
    regions = []
    for key in state_dict.keys():
        if "input" in key:
            remove_keys.append(key)
        if "target" in key:  # get all trained regions to initialise MaskTransformer
            regions.append(key.split("_")[0])
        if "target" in key and "target_guide" in key:  # get all trained regions to initialise MaskTransformer
            regions.append(key.split("_")[0])

    regions = list(dict.fromkeys(regions))

    for key in remove_keys:
        state_dict.pop(key)

    local_weights_available = state_dict is not None
    if local_weights_available:
        transformer = load(regions)
        if isinstance(transformer, paper.MaskMSTTransformer.__class__):
            if regions is not None:
                for region in regions:
                    init_image = torch.rand(state_dict[f"{region}_target_guide"].size())
                    init_guide = torch.ones(state_dict[f"{region}_target_guide"].size())
                    transformer.set_target_image(init_image, region)
                    transformer.set_target_guide(init_guide, region)
        else:
            init_image = torch.rand([1,1,512,512])
            transformer.set_target_image(init_image)
        transformer.load_state_dict(state_dict)
        return transformer
    else:
        return None



def load_local_weights(root, model_name):
    return torch.load(path.join(root, model_name))


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    dataset_dir = None
    model_dir = None
    device = None
    instance_norm = True
    masked = True
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
        image_results_dir = path.join(here, "data", "images", "results", "stylization")
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
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    for state in [False]:
        for device in ['cuda',]: # 'cpu'
            here = path.dirname(__file__)
            args.masked = state
            dataset_path = path.join(here, "data", "images", "dataset", "CelebAMask-HQ")
            # dataset_path = '~/datasets/celebamask/CelebAMask-HQ/'
            args.dataset_dir = (
                path.join(dataset_path, "CelebAMask-HQ-mask")
                if args.masked
                else path.join(
                    dataset_path, "CelebA-HQ-img"
                )
            )
            args.device = device
            main(args)