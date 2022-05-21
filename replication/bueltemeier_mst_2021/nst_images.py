import os
from argparse import Namespace
from os import path

import pystiche_papers.bueltemeier_mst_2021 as paper
from pystiche import misc, optim, loss, enc, ops, image


def nst(args):
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(weights="caffe", internal_preprocessing=True, allow_inplace=True)

    # content loss
    content_layer = "relu4_2"
    content_encoder = multi_layer_encoder.extract_encoder(content_layer)
    content_weight = 1e0
    content_loss = ops.FeatureReconstructionOperator(content_encoder, score_weight=content_weight)

    # style loss
    style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1",)
    style_weight = 1e3

    def get_style_op(encoder, layer_weight):
        return ops.GramOperator(encoder, score_weight=layer_weight)
    style_loss = ops.MultiLayerEncodingOperator(multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight)

    perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(args.device)

    images = paper.nst_images()
    images.download(args.image_source_dir)

    contents = ["bird"]  # ["mountain", "flower", "bird"]
    styles = ["mosaic2"]  # ["mosaic", "sky", "mosaic2", "abstract", "abstract2"]

    for content in contents:
        for style in styles:
            content_image = images[content].read(
                size=500, device=args.device
            )
            style_image = images[style].read(
                size=500, device=args.device
            )
            perceptual_loss.set_content_image(content_image)
            perceptual_loss.set_style_image(style_image)

            starting_point = "content"
            input_image = misc.get_input_image(starting_point, content_image=content_image)

            output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=500)

            output_file = path.join(
                args.image_results_dir, f"nst_IST_paper_{content}__{style}__full.jpg"
            )
            image.write_image(output_image, output_file)


def layer_nst(args):
    content = "bird"
    style = "mosaic2"

    images = paper.nst_images()
    images.download(args.image_source_dir)

    content_image = images[content].read(
        size=500, device=args.device
    )
    style_image = images[style].read(
        size=500, device=args.device
    )

    # style loss
    style_layers = ["relu1_1", "relu2_2", "relu3_1", "relu4_1",]

    for layer in style_layers:
        multi_layer_encoder = enc.vgg19_multi_layer_encoder(weights="caffe", internal_preprocessing=True,
                                                            allow_inplace=True)
        # content loss
        content_layer = "relu4_2"
        content_encoder = multi_layer_encoder.extract_encoder(content_layer)
        content_weight = 1e0
        content_loss = ops.FeatureReconstructionOperator(content_encoder, score_weight=content_weight)

        style_encoder = multi_layer_encoder.extract_encoder(layer)
        style_weight = 1e3
        style_loss = ops.GramOperator(style_encoder, score_weight=style_weight)

        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(args.device)
        perceptual_loss.set_content_image(content_image)
        perceptual_loss.set_style_image(style_image)

        starting_point = "content"
        input_image = misc.get_input_image(starting_point, content_image=content_image)

        output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=500)

        output_file = path.join(
            args.image_results_dir, f"nst_IST_paper_{content}__{style}__{layer}.jpg"
        )
        image.write_image(output_image, output_file)


def guided_nst(args):
    #######################################nst##########################################################################
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(weights="caffe", internal_preprocessing=True,
                                                        allow_inplace=True)
    # content loss
    content_layer = "relu4_2"
    content_encoder = multi_layer_encoder.extract_encoder(content_layer)
    content_weight = 1e0
    content_loss = ops.FeatureReconstructionOperator(content_encoder, score_weight=content_weight)

    # style loss
    style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    style_weight = 1e3

    def get_style_op(encoder, layer_weight):
        return ops.GramOperator(encoder, score_weight=layer_weight)

    style_loss = ops.MultiLayerEncodingOperator(multi_layer_encoder, style_layers, get_style_op,
                                                score_weight=style_weight)

    perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(args.device)

    images = paper.guide_nst_images(args.image_source_dir)

    content_image = images["landscape"].read(
        size=500, device=args.device
    )
    style_image = images["starry_night"].read(
        size=500, device=args.device
    )
    perceptual_loss.set_content_image(content_image)
    perceptual_loss.set_style_image(style_image)

    starting_point = "content"
    input_image = misc.get_input_image(starting_point, content_image=content_image)

    output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=500)

    output_file = path.join(
        args.image_results_dir, f"nst_IST_paper_guided__without.jpg"
    )
    image.write_image(output_image, output_file)

    #######################################guided nst##################################################################

    content_guides = images["landscape"].guides.read(size=500, device=args.device)

    style_guides = images["starry_night"].guides.read(size=500, device=args.device)

    regions = ("sky", "surroundings")

    def get_region_op(region, region_weight):
        return ops.MultiLayerEncodingOperator(
            multi_layer_encoder, style_layers, get_style_op, score_weight=region_weight,
        )

    style_loss = ops.MultiRegionOperator(regions, get_region_op, score_weight=style_weight)

    perceptual_loss = loss.GuidedPerceptualLoss(content_loss, style_loss).to(args.device)
    perceptual_loss.set_content_image(content_image)
    for region in regions:
        perceptual_loss.set_content_guide(region, content_guides[region])
        perceptual_loss.set_style_image(region, style_image)
        perceptual_loss.set_style_guide(region, style_guides[region])

    starting_point = "content"

    input_image = misc.get_input_image(starting_point, content_image=content_image)

    output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=500)

    output_file = path.join(
        args.image_results_dir, f"nst_IST_paper_guided__with.jpg"
    )
    image.write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    device = None

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "data", "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "data", "images", "results", "nst")
    image_results_dir = process_dir(image_results_dir)

    device = misc.get_device(device=device)
    logger = optim.OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    args = parse_input()
    nst(args)
    layer_nst(args)
    guided_nst(args)