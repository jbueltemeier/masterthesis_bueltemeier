import os
from os import path
import cv2
import numpy as np
import torch
from pystiche.image import verify_guides

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

here = path.dirname(__file__)

folder_mask_image = path.join(here, 'data', 'images', 'dataset', 'CelebAMask-HQ', 'CelebA-HQ-img')
folder_mask_base = path.join(here, 'data', 'images', 'dataset', 'CelebAMask-HQ', 'CelebAMask-HQ-mask-anno')
folder_save = path.join(here, 'data', 'images', 'dataset','CelebAMask-HQ', 'CelebAMask-HQ-mask')
img_num = 30000


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def remove_overlapping_pixels(base, ignore_labels):
    for label in ignore_labels:
        filename = os.path.join(folder_mask_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            base[im != 0] = 0

    return base


make_folder(folder_save)

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
                 'cloth'])

        if key == 'hair':
            im_base = remove_overlapping_pixels(
                im_base,
                ['l_brow', 'r_brow', 'l_ear', 'r_ear', 'cloth', 'eye_g', 'ear_r', 'neck_l']
            )

        if key == 'brows':
            im_base = remove_overlapping_pixels(
                im_base,
                ['eye_g', 'l_ear', 'r_ear', 'ear_r', 'neck_l', 'cloth']
            )

        if key == 'nose':
            im_base = remove_overlapping_pixels(
                im_base,
                ['eye_g', 'hair', 'l_brow', 'r_brow', 'hat', 'cloth', 'ear_r', 'neck_l', 'l_ear', 'r_ear']
            )

        if key == 'lips':
            im_base = remove_overlapping_pixels(
                im_base,
                ['hair', 'l_brow', 'r_brow', 'nose', 'ear_r', 'neck_l', 'cloth', 'l_ear', 'r_ear', 'hat']
            )

        if key == 'accessoire':
            im_base = remove_overlapping_pixels(
                im_base,
                ['l_ear', 'r_ear', 'cloth', 'eye_g']
            )

        if key == 'headwear':
            im_base = remove_overlapping_pixels(
                im_base,
                ['hair', 'eye_g', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'cloth', 'ear_r', 'neck_l']
            )

        if key == 'eye':
            im_base = remove_overlapping_pixels(
                im_base,
                ['hair', 'cloth', 'eye_g', 'nose', 'l_brow', 'r_brow', 'hat', 'ear_r', 'neck_l', 'l_ear', 'r_ear']
            )

        if key == 'ears':
            im_base = remove_overlapping_pixels(
                im_base,
                ['cloth', 'eye_g']
            )
        if key == 'glasses':
            im_base = remove_overlapping_pixels(
                im_base,
                ['cloth',]
            )


        if np.any(im_base != 0):
            filename_save = os.path.join(folder_save, str(k).rjust(5, '0'), 'guides', key + '.png')
            cv2.imwrite(filename_save, im_base)
            guides[key] = torch.from_numpy(im_base)


    print(verify_guides(guides))