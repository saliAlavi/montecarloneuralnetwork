import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import numpy as np
from PIL import Image

class AugmentImagePoints(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        #image, keypoints = sample['image'], sample['points']
        image, keypoints, valids = sample
        im = np.array(image)
        kpt = KeypointsOnImage([Keypoint(x, y) for x, y in keypoints], shape=im.shape)

        seq = iaa.Sequential([
            iaa.Multiply((1.2, 1.5)),
            iaa.Affine(scale=(0.8, 1.2), translate_px={"x": (-20, 20), "y": (-20, 20)}, rotate=(5, 10), shear=(1, 3)),
            iaa.LinearContrast((0.8, 1.2))
        ])

        img_aug, kpt_aug = seq(image=im, keypoints=kpt)
        keypoints_aug = None
        for i, point in enumerate(kpt_aug):
            if i == 0:
                keypoints_aug = [[point.x, point.y]]
            else:
                keypoints_aug = np.append(keypoints_aug, [[point.x, point.y]], axis=0)
        img_aug=Image.fromarray(img_aug)
        keypoints_aug=np.array(keypoints_aug)
        return img_aug, keypoints_aug,valids
