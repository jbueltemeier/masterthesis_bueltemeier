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
        "background": (style_images[style][0].to(args.device), style_images[style][1]["background"].to(args.device)),
        "skin": (style_images[style][0].to(args.device), style_images[style][1]["skin"].to(args.device)),
        "nose": (style_images[style][0].to(args.device), style_images[style][1]["nose"].to(args.device)),
        "glasses": (
            style_images["LRD_50_2008"][0].to(args.device),
            style_images["LRD_50_2008"][1]["accessoire"].to(args.device),
        ),
        "eye": (style_images[style][0].to(args.device), style_images[style][1]["eye"].to(args.device)),
        "brows": (style_images[style][0].to(args.device), style_images[style][1]["brows"].to(args.device)),
        "ears": (style_images[style][0].to(args.device), style_images[style][1]["ears"].to(args.device))if style not in ["Specimen_0_2",
        "Specimen_0_2005"] else (style_images["MAD_20_2005"][0].to(args.device), style_images["MAD_20_2005"][1]["eye"].to(args.device)),
        "lips": (style_images[style][0].to(args.device), style_images[style][1]["lips"].to(args.device)),
        "hair": (style_images[style][0].to(args.device), style_images[style][1]["hair"].to(args.device)),
        "headwear": (
            style_images["MAD_2000_2002"][0].to(args.device),
            style_images["MAD_2000_2002"][1]["headwear"].to(args.device),
        ),
        "accessoire": (
            style_images["GBP_5_2002"][0].to(args.device),
            style_images["GBP_5_2002"][1]["accessoire"].to(args.device),
        ),
        "body": (style_images["MAD_20_2005"][0].to(args.device), style_images["MAD_20_2005"][1]["body"].to(args.device)),
    }