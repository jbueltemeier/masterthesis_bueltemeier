import os
from argparse import Namespace
from os import path
# from memory_profiler import profile
import torch
from pystiche import misc, optim, image
import pystiche_papers.bueltemeier_mst_2021 as paper
import utils as _utils
from pystiche_papers.bueltemeier_mst_2021._utils import hyper_parameters as _hyper_parameters

def segmentation_image(content_guides, output_name):
    segmentaion_image = image.guides_to_segmentation(
        content_guides,
        color_map=_utils.seg_color_map
    )
    output_file = path.join(args.image_results_dir, f"{output_name}_seg.png")
    image.write_image(segmentaion_image, output_file)


def stylise_images(
        dataset,
        transformer,
        model_name,
        masked,
        save_segmentation=False,
        time_track=False,
        number_images=0):
    times = []
    if masked:
        for i in range(number_images):
            content_image, content_guides = dataset.__getitem__(i)
            content_image = content_image.to(args.device)
            for key, guide in content_guides.items():
                content_guides[key] = content_guides[key].to(args.device)
            output_image, excecution_time = paper.mask_stylization(content_image,
                                                                   content_guides,
                                                                   transformer,
                                                                   track_time=time_track)
            if excecution_time is not None:
                times.append(excecution_time)

            style = model_name.split("__")[2]
            output_name = f"intaglio_mask_{i}_{style}__{args.device}"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)

            # save segementation image
            if save_segmentation:
                segmentation_image(content_guides, output_name)
        else:
            for i in range(number_images):
                content_image = dataset.__getitem__(i)
                content_image = content_image.to(args.device)
                output_image, excecution_time = paper.stylization(content_image,
                                                                  transformer,
                                                                  track_time=time_track)

                if excecution_time is not None:
                    times.append(excecution_time)

                style = model_name.split("__")[1]
                output_name = f"intaglio_{i}_{style}__{args.device}"
                if args.instance_norm:
                    output_name += "__instance_norm"
                output_file = path.join(args.image_results_dir, f"{output_name}.png")
                image.write_image(output_image, output_file)

        if times:
            # run all operations once for cuda warm-up -> time error first item
            times = times[1:]
            message = f"{args.device} time: {sum(times) / (number_images - 1)}"
            print(message)


def save_content_image(content_image, output_name):
    output_file = path.join(args.image_results_dir, f"{output_name}_image.png")
    image.write_image(content_image, output_file)


def whitening_background(output_image, content_guides, output_name):
    output_file = path.join(
        args.image_results_dir,
        f"{output_name}_background.png"
    )
    masked_image = output_image.masked_fill(content_guides["background"] == 1, 1)
    image.write_image(masked_image, output_file)


def stylise_image_numbers(
        transformer,
        masked,
        save_segmentation=True,
        content_image_save=True,
        white_background=True
):
    for image_number in _utils.image_numbers:
        content_image, content_guides = _utils.get_guided_images_from_dataset(
            args,
            image_number
        )
        if masked:
            output_image, _ = paper.mask_stylization(
                content_image,
                content_guides,
                transformer
            )
        else:
            output_image, _ = paper.stylization(content_image, transformer)
        output_name = f"{image_number}_intaglio"
        if args.instance_norm:
            output_name += "__instance_norm"
        if masked:
            output_name += "__mask"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)

        if save_segmentation and masked:
            segmentation_image(content_guides, output_name)

        if content_image_save:
            save_content_image(content_image, output_name)

        if white_background:
            whitening_background(output_image, content_guides, output_name)


def stylise_detail_images(transformer, image_size, masked=True, save_image=False):
    hyper_parameters = init_detail_hyper_parameters(image_size)
    for image_number, detail_position in _utils.detail_image_numbers:
        content_image, content_guides = _utils.get_guided_images_from_dataset(
            args, image_number)
        content_image = _utils.crop_image_detail(content_image, detail_position)
        content_guides = _utils.crop_guides_detail(content_guides, detail_position)
        if masked:
            output_image, _ = paper.mask_stylization(
                content_image,
                content_guides,
                transformer,
                hyper_parameters=hyper_parameters
            )
        else:
            output_image, _ = paper.stylization(content_image, transformer)
        regions = ''.join(list(content_guides.keys()))
        output_name = f"{image_number}_intaglio_detail_{regions}"
        if args.instance_norm:
            output_name += "__instance_norm"
        if masked:
            output_name += "__mask"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        if "background" in content_guides.keys():
            output_image = output_image.masked_fill(
                content_guides["background"] == 1,
                1
            )
        image.write_image(output_image, output_file)

        if save_image:
            transform = _utils.content_transform(170)
            output_file = path.join(args.image_results_dir, f"{output_name}_image.png")
            image.write_image(transform(content_image), output_file)


def stylise_main(args, time_track=False, detail=True, save_segmentation=True):
    if args.masked:
        model_name = "bueltemeier_2021__mask__UHD_20_1997__intaglio__instance_norm-e12b57fd.pth"
        # model_name = "bueltemeier_2021__mask__MAD_20_2005__intaglio__instance_norm-ba8a81c1.pth"
        transformer = load_transformer(
            path.join(args.model_dir, "stylization"),
            args.instance_norm,
            model_name
        )
        dataset = paper.mask_dataset(path.join(args.dataset_dir))
        stylise_images(
            dataset,
            transformer,
            model_name,
            args.masked,
            time_track=time_track,
            save_segmentation=save_segmentation,
            number_images=0
        )
        stylise_image_numbers(
            transformer,
            args.masked,
            save_segmentation=save_segmentation,
            content_image_save=True,
            white_background=True,
        )

        if detail:
            stylise_detail_images(transformer, masked=args.masked)
    else:
        model_name = "bueltemeier_2021__MAD_20_2005__intaglio__instance_norm-379670d4.pth"
        transformer = load_transformer(
            path.join(args.model_dir, "stylization"),
            args.instance_norm,
            model_name
        )
        dataset = paper.dataset(path.join(args.dataset_dir))
        stylise_images(
            dataset,
            transformer,
            model_name,
            args.masked,
            time_track=time_track,
            save_segmentation=save_segmentation,
            number_images=0
        )
        stylise_image_numbers(
            transformer,
            args.masked,
            save_segmentation=save_segmentation,
            content_image_save=True,
            white_background=True,
        )
        if detail:
            stylise_detail_images(transformer, masked=args.masked)


def binary_edge_stylise_image_numbers(
        transformer,
        save_segmentation=True,
        content_image_save=True,
        white_background=True
):
    for image_number in _utils.image_numbers:
        content_image, content_guides = _utils.get_guided_images_from_dataset(
            args,
            image_number
        )
        result_images = []
        for region, guide in content_guides.items():
            output_image, _ = paper.mask_stylization(
                content_image,
                {region: guide},
                transformer,
            )
            output_image = output_image * guide
            result_images.append(output_image)
        output_image = torch.sum(torch.stack(result_images), dim=0)
        output_name = f"{image_number}_intaglio"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_name += "__mask__binaryEdge"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        image.write_image(output_image, output_file)

        if save_segmentation:
            segmentation_image(content_guides, output_name)

        if content_image_save:
            save_content_image(content_image, output_name)

        if white_background:
            whitening_background(output_image, content_guides, output_name)


def init_detail_hyper_parameters(image_size):
    hyper_parameters = _hyper_parameters()
    hyper_parameters.content_transform.image_size = image_size
    hyper_parameters.style_transform.image_size = image_size
    return hyper_parameters


def binary_edge_substyle_stylisation(transformer, image_size,save_segmentation=True, save_image=True):
    hyper_parameters = init_detail_hyper_parameters(image_size)
    for image_number, detail_position in _utils.detail_image_numbers:
        content_image, content_guides = _utils.get_guided_images_from_dataset(
            args, image_number)
        content_image = _utils.crop_image_detail(content_image, detail_position)
        content_guides = _utils.crop_guides_detail(content_guides, detail_position)
        result_images = []
        for region, guide in content_guides.items():
            output_image, _ = paper.mask_stylization(
                content_image,
                {region: guide},
                transformer,
                hyper_parameters=hyper_parameters
            )
            transform = _utils.content_mask_transform(image_size)
            output_image = output_image * transform(guide)
            result_images.append(output_image)
        output_image = torch.sum(torch.stack(result_images), dim=0)

        regions = ''.join(list(content_guides.keys()))
        output_name = f"{image_number}_intaglio_detail_{regions}"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_name += "__mask__binaryEdge"
        output_file = path.join(args.image_results_dir, f"{output_name}.png")
        if "background" in content_guides.keys():
            output_image = output_image.masked_fill(
                transform(content_guides["background"]) == 1,
                1
            )
        image.write_image(output_image, output_file)

        if save_segmentation:
            segmentation_image(content_guides, output_name)
        if save_image:
            output_file = path.join(args.image_results_dir, f"{output_name}_image.png")
            image.write_image(content_image, output_file)


def edge_stylisation(args):
    model_name = "bueltemeier_2021__mask__UHD_20_1997__intaglio__instance_norm-e12b57fd.pth"
    # model_name = "bueltemeier_2021__mask__MAD_20_2005__intaglio__instance_norm-ba8a81c1.pth"
    transformer = load_transformer(
        path.join(args.model_dir, "stylization"),
        args.instance_norm,
        model_name
    )

    stylise_image_numbers(
        transformer,
        args.masked,
        save_segmentation=False,
        content_image_save=True,
        white_background=True,
    )
    binary_edge_stylise_image_numbers(transformer)

    # image_size = 200
    # stylise_detail_images(transformer, image_size, masked=args.masked)
    # binary_edge_substyle_stylisation(transformer, image_size, save_image=False)


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
        if "input_guide" in key:
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
                        image_size = _utils.UHD_20_1997[region]
                        init_image = torch.rand(image_size)
                        init_guide = torch.ones(image_size)
                        state_dict[f"{region}_target_guide"] = init_guide
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
        dataset = "CelebAMask-HQ-mask" if masked else "CelebA-HQ-img"
        dataset_dir = path.join(  # default to masked
            here,
            "data",
            "images",
            "dataset",
            "CelebAMask-HQ",
            dataset
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
        programming_dataset=True,  # switch dataset from intern to extern (Rechenknecht)
    )


def stylisation(args, track_time=False):
    for masked_state in [True, False]:
        if track_time:
            for device in ['cuda', 'cpu']:
                args.masked = masked_state
                args.dataset_dir = _utils.init_dataset(
                    masked=args.masked,
                    programming_dataset=True
                )
                args.device = device
                stylise_main(args)
        else:
            args.masked = masked_state
            args.dataset_dir = _utils.init_dataset(
                masked=args.masked,
                programming_dataset=True
            )
            stylise_main(args)


if __name__ == "__main__":
    args = parse_input()
    # stylisation(args)
    edge_stylisation(args)
