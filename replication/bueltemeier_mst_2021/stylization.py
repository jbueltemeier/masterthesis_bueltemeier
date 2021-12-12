import os
from argparse import Namespace
from os import path
from memory_profiler import profile

import torch
from pystiche import misc, optim, image
import pystiche_papers.bueltemeier_mst_2021 as paper

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

def main(args, time_track = False):
    n = 10
    if args.masked:
        # model_name = "bueltemeier_2021__mask__DM_100_1996__intaglio__instance_norm-fccc49fd.pth"
        # model_name = "bueltemeier_2021__mask__MAD_20_2005__intaglio__instance_norm-86e4c28e.pth"
        # model_name = "bueltemeier_2021__mask__Specimen_0_2__intaglio__instance_norm-d6f9b1ff.pth"
        model_name = "bueltemeier_2021__mask__UHD_20_1997__intaglio__instance_norm-8ddf3661.pth"
        #
        # model_name = "bueltemeier_2021__mask__DM_100_1996__intaglio__instance_norm-03045821.pth"
        # model_name = "bueltemeier_2021__mask__UHD_20_1997__intaglio__instance_norm-e12b57fd.pth"
        transformer = load_transformer(args.model_dir, args.instance_norm, model_name)

        dataset = paper.mask_dataset(path.join(args.dataset_dir))
        times = []
        for i in range(n):
            content_image, content_guides = dataset.__getitem__(i)
            content_image = content_image.to(args.device)
            for key, guide in content_guides.items():
                content_guides[key] = content_guides[key].to(args.device)
            output_image, excecution_time = paper.mask_stylization(content_image, content_guides, transformer, track_time=time_track)

            if excecution_time is not None:
                times.append(excecution_time)

            style = model_name.split("__")[2]
            output_name = f"intaglio_mask_{i}_{style}__{args.device}"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)

            # save segementation image
            segmentaion_image = image.guides_to_segmentation(content_guides, color_map=seg_color_map)
            output_file = path.join(args.image_results_dir, f"{output_name}_seg.png")
            image.write_image(segmentaion_image, output_file)


        if times:
            times = times[1:] # run all operations once for cuda warm-up -> time error first item
            message = f"{args.device} time: {sum(times) / (n-1)}"
            print(message)
    else:
        # model_name = "bueltemeier_2021__MAD_20_2005__intaglio__instance_norm-379670d4.pth"
        #
        # model_name = "bueltemeier_2021__DM_100_1996__intaglio__instance_norm-09995fd0.pth"
        model_name = "bueltemeier_2021__UHD_20_1997__intaglio__instance_norm-9007c143.pth"
        transformer = load_transformer(args.model_dir, args.instance_norm, model_name)
        dataset = paper.dataset(path.join(args.dataset_dir))

        times = []
        for i in range(n):
            content_image = dataset.__getitem__(i)
            content_image = content_image.to(args.device)
            output_image, excecution_time = paper.stylization(content_image, transformer, track_time=time_track)

            if excecution_time is not None:
                times.append(excecution_time)

            style = model_name.split("__")[1]
            output_name = f"intaglio_{i}_{style}__{args.device}"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)

        if times:
            times = times[1:] # run all operations once for cuda warm-up -> time error first item
            message = f"{args.device} time: {sum(times) / (n-1)}"
            print(message)


def load_transformer(model_dir, instance_norm, model_name):
    def load(regions=None):
        if len(regions) > 1:
            return paper.MaskMSTTransformer(regions, in_channels=1, instance_norm=instance_norm, recalc_enc=False)
        else:
            return paper.MSTTransformer(in_channels=1, instance_norm=instance_norm)

    state_dict = load_local_weights(model_dir, model_name)

    regions = []
    remove_keys = []
    for key in state_dict.keys():
        if "target" in key:  # get all trained regions to initialise MaskTransformer
            regions.append(key.split("_")[0])
        if "target_guide" in key:
            remove_keys.append(key)

    for key in remove_keys:
         del state_dict[key]
    regions = list(dict.fromkeys(regions))

    local_weights_available = state_dict is not None
    if local_weights_available:
        transformer = load(regions)
        if isinstance(transformer, paper.MaskMSTTransformer):
            if regions is not None:
                for region in regions:
                    if f"{region}_target_guide" in state_dict.keys():
                        init_image = torch.rand(state_dict[f"{region}_target_guide"].size())
                        init_guide = torch.ones(state_dict[f"{region}_target_guide"].size())
                    else:
                        image_size = UHD_20_1997[region]
                        init_image = torch.rand(image_size)
                        init_guide = torch.ones(image_size)
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
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    for state in [True]:
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