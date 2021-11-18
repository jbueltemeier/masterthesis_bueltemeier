import os
from os import path
import cv2
import numpy as np
import torch
from pystiche.image import verify_guides
from argparse import Namespace

label_list = {
    'background': ['skin', 'neck', 'nose', 'eye_g',
                   'l_eye', 'r_eye', 'l_brow', 'r_brow',
                   'l_ear', 'r_ear','mouth', 'u_lip', 'l_lip',
                   'hair', 'hat', 'ear_r', 'neck_l', 'cloth'],
    'skin': ['skin', 'neck'],
    'nose': ['nose'],
    'glasses': ['eye_g'],
    'eye': ['l_eye', 'r_eye'],
    'brows': ['l_brow', 'r_brow'],
    'ears': ['l_ear', 'r_ear'],
    'lips': ['mouth', 'u_lip', 'l_lip'],
    'hair': ['hair'],
    'headwear': ['hat'],
    'accessoire': ['ear_r', 'neck_l'],
    'body': ['cloth'],
}


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def remove_overlapping_pixels(base, ignore_labels, folder_mask_base, folder_num, k):
    for label in ignore_labels:
        filename = os.path.join(folder_mask_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            base[im != 0] = 0

    return base


def main(args):
    folder_path = args.dataset_dir
    folder_mask_image = path.join(folder_path,  'CelebA-HQ-img')
    folder_mask_base = path.join(folder_path,  'CelebAMask-HQ-mask-anno')
    folder_save = path.join(folder_path, 'CelebAMask-HQ-mask')
    make_folder(folder_save)
    img_num = 30000

    for k in range(img_num):
        folder_num = k // 2000
        guides = {}
        for idx, key in enumerate(label_list.keys()):
            make_folder(path.join(folder_save, str(k).rjust(5, '0')))
            make_folder(path.join(folder_save, str(k).rjust(5, '0'), 'guides'))
            filename = os.path.join(folder_mask_image, str(k) + '.jpg')
            if (os.path.exists(filename)):
                im = cv2.imread(filename)
                filename_save = path.join(folder_save,str(k).rjust(5, '0'), str(k) + '.jpg')
                cv2.imwrite(filename_save, im)
            im_base = np.zeros((512, 512))
            for label in label_list[key]:
                filename = os.path.join(folder_mask_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                if (os.path.exists(filename)):
                    im = cv2.imread(filename)
                    im = im[:, :, 0]
                    im_base[im != 0] = 255

            if key == 'background':  # invert background mask
                im_base_save = np.ones((512, 512))
                im_base_save[im_base == 0] = 255
                im_base_save[im_base == 255] = 0
                im_base_save[im_base_save == 1] = 255
                im_base = im_base_save

            # remove overlap regions in CelebAMask-HQ regions
            if key == 'skin':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',  'l_ear',
                     'r_ear','mouth', 'u_lip', 'l_lip','hair', 'hat', 'ear_r', 'neck_l',
                     'cloth'], folder_mask_base, folder_num, k)

            if key == 'hair':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['l_brow', 'r_brow', 'l_ear', 'r_ear', 'cloth', 'eye_g', 'ear_r', 'neck_l'],
                folder_mask_base, folder_num, k
                )

            if key == 'brows':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['eye_g', 'l_ear', 'r_ear', 'ear_r', 'neck_l', 'cloth'], folder_mask_base, folder_num, k
                )

            if key == 'nose':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['eye_g', 'hair', 'l_brow', 'r_brow', 'hat', 'cloth', 'ear_r', 'neck_l', 'l_ear', 'r_ear'], folder_mask_base, folder_num, k
                )

            if key == 'lips':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['hair', 'l_brow', 'r_brow', 'nose', 'ear_r', 'neck_l', 'cloth', 'l_ear', 'r_ear', 'hat'], folder_mask_base, folder_num, k
                )

            if key == 'accessoire':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['l_ear', 'r_ear', 'cloth', 'eye_g'], folder_mask_base, folder_num, k
                )

            if key == 'headwear':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['hair', 'eye_g', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'cloth', 'ear_r', 'neck_l'], folder_mask_base, folder_num, k
                )

            if key == 'eye':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['hair', 'cloth', 'eye_g', 'nose', 'l_brow', 'r_brow', 'hat', 'ear_r', 'neck_l', 'l_ear', 'r_ear'], folder_mask_base, folder_num, k
                )

            if key == 'ears':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['cloth', 'eye_g'], folder_mask_base, folder_num, k
                )
            if key == 'glasses':
                im_base = remove_overlapping_pixels(
                    im_base,
                    ['cloth',], folder_mask_base, folder_num, k
                )


            if np.any(im_base != 0):
                filename_save = os.path.join(folder_save, str(k).rjust(5, '0'), 'guides', key + '.png')
                cv2.imwrite(filename_save, im_base)
                guides[key] = torch.from_numpy(im_base)

        if (k % 100) == 0:
            print(k)

        if k == 0:
            print(k)
            print(folder_save)


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
    dataset_dir = 'C:\\Users\\julia\\Downloads\\CelebAMask-HQ'
    dataset_dir = process_dir(dataset_dir)

    return Namespace(
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    args = parse_input()
    main(args)
