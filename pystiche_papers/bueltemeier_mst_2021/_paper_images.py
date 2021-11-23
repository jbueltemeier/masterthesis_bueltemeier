from os import path

from pystiche import data


__all__ = [
    "nst_images",
    "guide_nst_images",
]

def nst_images():
    images_ = {
        "sky": data.DownloadableImage(
            "https://free-images.com/md/2397/abstract_astrology_astronomy_315181.jpg",
            license=data.PublicDomainLicense(),
            md5="87ac575a293d7d669874b43070675b2d",
            file="abstract_astrology_astronomy_315181.jpg",
        ),
        "mosaic": data.DownloadableImage(
            "https://free-images.com/md/1278/mosaic_ducks_massimo.jpg",
            license=data.PublicDomainLicense(),
            md5="4d429a89f9d7a62f94e87e1f607f221c",
            file="mosaic_ducks_massimo.jpg",
        ),
        "mosaic2": data.DownloadableImage(
            "https://free-images.com/md/ff61/mosaic_color_colorful_pattern.jpg",
            license=data.PublicDomainLicense(),
            md5="ce766692a6048d6c9ea8385bc08cd543",
            file="mosaic_color_colorful_pattern.jpg",
        ),
        "abstract": data.DownloadableImage(
            "https://free-images.com/md/5b7f/abstract_art_colorful_1085200.jpg",
            license=data.PublicDomainLicense(),
            md5="efa658b17912e01a9685c68c4fc64aec",
            file="abstract_art_colorful_1085200.jpg",
        ),
        "abstract2": data.DownloadableImage(
            "https://free-images.com/md/1f5a/art_color_abstract_design.jpg",
            license=data.PublicDomainLicense(),
            md5="63dce4b139cd71e119a938a7ba6f68d7",
            file="art_color_abstract_design.jpg",
        ),
        "bird": data.DownloadableImage(
            "https://free-images.com/lg/2b9d/bird_wildlife_sky_clouds.jpg",
            license=data.PublicDomainLicense(),
            md5="79da0d0b491678e34360973335147a17",
            file="bird_wildlife_sky_clouds.jpg",
        ),
        "flower": data.DownloadableImage(
            "https://free-images.com/lg/19a7/little_orchid_d1212_light.jpg",
            license=data.PublicDomainLicense(),
            md5="d252e56077a239c7eed18e31cbff0ac3",
            file="little_orchid_d1212_light.jpg",
        ),
        "mountain": data.DownloadableImage(
            "https://free-images.com/md/ae78/landscape_nature_sky_clouds.jpg",
            license=data.PublicDomainLicense(),
            md5="ac7e753b0226929a07afc674abeb8a3b",
            file="landscape_nature_sky_clouds.jpg",
        ),
        "shipwreck": data.DownloadableImage(
            "https://blog-imgs-51.fc2.com/b/e/l/bell1976brain/800px-Shipwreck_turner.jpg",
            title="Shipwreck of the Minotaur",
            author="J. M. W. Turner",
            date="ca. 1810",
            license=data.ExpiredCopyrightLicense(1851),
            md5="4fb76d6f6fc1678cb74e858324d4d0cb",
            file="shipwreck_of_the_minotaur__turner.jpg",
        ),
        "starry_night": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            title="Starry Night",
            author="Vincent van Gogh",
            date="ca. 1889",
            license=data.ExpiredCopyrightLicense(1890),
            md5="372e5bc438e3e8d0eb52cc6f7ef44760",
        ),
        "the_scream": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
            title="The Scream",
            author="Edvard Munch",
            date="ca. 1893",
            license=data.ExpiredCopyrightLicense(1944),
            md5="46ef64eea5a7b2d13dbadd420b531249",
        ),
        "femme_nue_assise": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg",
            title="Figure dans un Fauteuil",
            author="Pablo Ruiz Picasso",
            date="ca. 1909",
            license=data.ExpiredCopyrightLicense(1973),
            md5="ba14b947b225d9e5c59520a814376944",
        ),
        "composition_vii": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
            title="Composition VII",
            author="Wassily Kandinsky",
            date="1913",
            license=data.ExpiredCopyrightLicense(1944),
            md5="bfcbc420684bf27d2d8581fa8cc9522f",
        ),
    }
    return data.DownloadableImageCollection(images_)


def guide_nst_images(root: str):
    local_path = path.join(root, "guided")
    images_ = {
        "cliff": data.LocalImage(
            file=path.join(local_path, "cliff_walk.jpg"),
            collect_local_guides=True
        ),
        "landscape": data.LocalImage(
            file=path.join(local_path, "landscape_nature_sky_clouds.jpg"),
            collect_local_guides=True
        ),
        "starry_night": data.LocalImage(
            file=path.join(local_path, "starry_night.jpg"),
            collect_local_guides=True
        ),
    }
    return data.LocalImageCollection({**images_},)