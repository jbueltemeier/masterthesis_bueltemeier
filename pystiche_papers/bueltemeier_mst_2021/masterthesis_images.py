from os import path
import torch
from pystiche import data, image, misc, enc, ops, loss, optim


region_map = {
        "brows": (247, 129, 191),
        "eye": (55, 126, 184),
        "nose": (77, 175, 74),
        "skin": (152, 78, 163),
        "hair": (255, 127, 0),
        "body": (228, 26, 28),
        "ears": (166, 86, 40),
        "lips": (255, 255, 51),
        "headwear": (140, 255, 251),
        "background": (153, 153, 153),
        "accessoire": (255, 174, 200),
        "beard": (255, 255, 255),
        "wheatfield": (195,195,195),
        "sky": (0,0,0),
    }


def images(local_path):
    nst_path = path.join(local_path, "NST")
    images_ = {
        "wheatfield_style": data.LocalImage(
            file=path.join(nst_path, "Wheat_Field_with_Cypresses.jpg"),
            collect_local_guides=True
        ),
        "wheatfield": data.LocalImage(
            file=path.join(nst_path, "WheatField.jpg"),
            collect_local_guides=True
        ),
        "starry_night": data.LocalImage(
            file=path.join(nst_path, "starry_night.jpg"),
            collect_local_guides=False
        ),
        "auvers": data.LocalImage(
            file=path.join(nst_path, "Van_Gogh_Ebene_bei_Auvers.jpg"),
            collect_local_guides=True
        ),
        "DM": data.LocalImage(
            file=path.join(nst_path, "DM_100_1996.png"),
            collect_local_guides=True
        ),
        "Men": data.LocalImage(
            file=path.join(nst_path, "22555.png"),
            collect_local_guides=True
        ),
        "dog": data.LocalImage(
            file=path.join(nst_path, "content.jpg"),
            collect_local_guides=False
        ),
        "mosaic": data.LocalImage(
            file=path.join(nst_path, "style.jpg"),
            collect_local_guides=False
        ),
    }
    return data.LocalImageCollection(images_)


def create_save_name(save_name):
    local_path = "C:/Users/julia/Desktop/Arbeit/Masterarbeit/Masterarbeit/LaTex-Vorlage/images"
    return path.join(local_path, save_name)


def load_images():
    device = misc.get_device(device=None)
    local_path = "C:/Users/julia/Desktop/Arbeit/Masterarbeit/Masterarbeit/LaTex-Vorlage/images"
    return images(local_path), device


def nst_script(
    content_image,
    style_images,
    device,
    output_file,
    content_guides=None,
    style_guides=None,
    content_layer="relu4_2",
    content_weight=1e0,
    guided=False,
    style_layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1"),
    style_weight=1e2,
    starting_point="content",
    num_steps=500,
):
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(weights="caffe",
                                                        internal_preprocessing=True,
                                                        allow_inplace=True)
    # content loss
    content_encoder = multi_layer_encoder.extract_encoder(content_layer)
    content_loss = ops.FeatureReconstructionOperator(content_encoder,
                                                     score_weight=content_weight)

    # style loss
    def get_style_op(encoder, layer_weight):
        return ops.GramOperator(encoder, score_weight=layer_weight)
    if not guided:
        style_loss = ops.MultiLayerEncodingOperator(multi_layer_encoder, style_layers,
                                                    get_style_op,
                                                    score_weight=style_weight)
        perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(device)
        perceptual_loss.set_content_image(content_image)
        perceptual_loss.set_style_image(style_images)
    else:
        def get_region_op(region, region_weight):
            return ops.MultiLayerEncodingOperator(
                multi_layer_encoder, style_layers, get_style_op, score_weight=region_weight,
            )
        regions = list(style_images.keys())
        style_loss = ops.MultiRegionOperator(regions, get_region_op,
                                         score_weight=style_weight)

        perceptual_loss = loss.GuidedPerceptualLoss(content_loss, style_loss).to(device)
        perceptual_loss.set_content_image(content_image)
        for region in regions:
            perceptual_loss.set_content_guide(region, content_guides[region])
            perceptual_loss.set_style_image(region, style_images[region])
            perceptual_loss.set_style_guide(region, style_guides[region])

    input_image = misc.get_input_image(starting_point, content_image=content_image)
    output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=num_steps)
    image.write_image(output_image, output_file)


def iteration_nst():
    images_, device = load_images()
    content_image = images_["dog"].read(size=512, device=device)
    style_image = images_["mosaic"].read(size=512, device=device)
    for iteration in [10, 100, 500]:
        output_file = create_save_name(f"nst_iteration__{iteration}.jpg")
        nst_script(content_image, style_image, device, output_file, num_steps=iteration)

    output_file = create_save_name("nst_result.jpg")
    nst_script(content_image, style_image, device, output_file, style_weight=1e0)



def guided_nst():
    images_, device = load_images()
    image_size = 512

    content_image = images_["wheatfield"].read(size=image_size, device=device)
    style_image = images_["wheatfield_style"].read(size=image_size, device=device)

    # without guiding
    output_file = create_save_name("nst_guided__without.jpg")
    nst_script(
        content_image,
        style_image,
        device,
        output_file,
        style_layers=("relu1_1", "relu2_2", "relu3_1"),
        style_weight=1e1,
        num_steps=1000
    )

    # with guiding
    content_guides = images_["wheatfield"].guides.read(size=image_size, device=device)
    style_guides = images_["wheatfield_style"].guides.read(size=image_size, device=device)

    style_images = {
        "sky": style_image,
        "wheatfield": style_image
    }
    output_file = create_save_name("nst_guided__with.jpg")
    nst_script(
        content_image,
        style_images,
        device,
        output_file,
        guided=True,
        content_guides=content_guides,
        style_guides=style_guides,
        style_layers=("relu1_1", "relu2_2", "relu3_1"),
        style_weight=1e1,
        num_steps=1000
    )
    second_style_image = images_["starry_night"].read(size=image_size, device=device)
    style_images = {
        "sky":  second_style_image,
        "wheatfield": style_image
    }
    second_image_sizes = image.extract_image_size(second_style_image)
    style_guides_diff = {
        "sky":  torch.ones((1,1,second_image_sizes[0], second_image_sizes[1])).to(device),
        "wheatfield": style_guides["wheatfield"]
    }
    output_file = create_save_name("nst_guided__different_styles.jpg")
    nst_script(
        content_image,
        style_images,
        device,
        output_file,
        guided=True,
        content_guides=content_guides,
        style_guides=style_guides_diff,
        style_layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1",),
        style_weight=1e-3,
        num_steps=1000
    )


def generated_substyle_nst():
    images_, device = load_images()
    image_size = 512

    content_image = images_["wheatfield"].read(size=image_size, device=device)
    style_image = images_["wheatfield_style"].read(size=image_size, device=device)

    # complete style image style generation
    output_file = create_save_name(f"generated_style_nst_guided_complete.jpg")
    nst_script(
        content_image,
        style_image,
        device,
        output_file,
        content_weight=0,
        style_weight=1e1,
        num_steps=1500,
        starting_point="random",
    )

    regions = ("sky", "wheatfield")
    content_guides = {}
    for region in regions:
        content_image_size = image.extract_image_size(content_image)
        content_guides[region] = torch.ones((1,1,content_image_size[0],content_image_size[1])).to(device)

    style_guides = images_["wheatfield_style"].guides.read(size=image_size, device=device)

    for subregion in regions:
        output_file = create_save_name(f"generated_style_nst_guided_{subregion}.jpg")
        nst_script(
            content_image,
            {subregion: style_image},
            device,
            output_file,
            content_guides={subregion: content_guides[subregion]},
            style_guides={subregion: style_guides[subregion]},
            guided=True,
            content_weight=0,
            style_weight=1e0,
            num_steps=1500,
            starting_point="random",
        )


def create_masked_images_script(image_str: str="wheatfield"):
    images_, device = load_images()
    image_size = 512
    complete_image = images_[image_str].read(size=image_size, device=device)
    guides = images_[image_str].guides.read(size=image_size, device=device)

    for region, guide in guides.items():
        output_image = complete_image * guide
        output_file = create_save_name(f"guided_image__{image_str}__{region}.jpg")
        image.write_image(output_image, output_file)

    output_image = image.guides_to_segmentation(guides, region_map)
    output_file = create_save_name(f"{image_str}__seg.png")
    image.write_image(output_image, output_file)

def create_masked_images():
    for name in ["wheatfield", "wheatfield_style", "auvers"]:
        create_masked_images_script(name)


def create_images_masterthesis_fundamentals():
    iteration_nst()
    guided_nst()
    generated_substyle_nst()
    create_masked_images()
