import os
from os import path
from argparse import Namespace




def main(args):
    folder_path = args.dataset_dir
    dataset_folder = path.join(folder_path,  'CelebAMask-HQ-mask')
    img_num = 30000

    count_attributes = {
        'background': 0,
        'skin': 0,
        'nose': 0,
        'glasses': 0,
        'eye': 0,
        'brows': 0,
        'ears': 0,
        'lips': 0,
        'hair': 0,
        'headwear': 0,
        'accessoire': 0,
        'body': 0,
    }

    for k in range(img_num):
        guide_folder = path.join(dataset_folder, str(k).rjust(5, '0'), 'guides')
        guides_names = os.listdir(guide_folder)
        for guide_name in guides_names:
            attribute = guide_name.split(".")[0]
            count_attributes[attribute] += 1

    for attribute, count in count_attributes.items():
        print(
            f"There is a mask for the attribute {attribute} in {count} of the images."
        )



def parse_input():
    # TODO: write CLI
    dataset_dir = None

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if dataset_dir is None:
        dataset_dir = (
            path.join(
                here, "data", "images", "dataset", "CelebAMask-HQ"
            )
        )
    dataset_dir = process_dir(dataset_dir)

    return Namespace(
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    args = parse_input()
    main(args)