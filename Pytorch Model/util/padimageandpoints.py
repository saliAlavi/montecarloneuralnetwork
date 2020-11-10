from skimage import io
from pathlib import Path
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from scipy.io import loadmat, savemat
import imageio

list_files = list(Path('../dataset/images').glob('*.jpg'))
images=[]
images_shape=[]
for files_name in list_files[:10]:
    image= io.imread(files_name)
    image=np.asarray(image)
    images_shape.append(image.shape)
    images.append(image)

h = max(images_shape,key=lambda x: x[0])[0]
w = max(images_shape,key=lambda x: x[1])[1]

aug = iaa.Sequential([
            iaa.PadToFixedSize(width=w, height=h),
            iaa.CropToFixedSize(width=w, height=h)
        ])

joints = loadmat('../dataset/joints.mat')
joints = joints['joints']

joints = np.transpose(joints, (2, 1, 0))
kpt_all=[]
for index, (joint,image) in enumerate(zip(joints, images)):
    kpt = KeypointsOnImage([Keypoint(x, y) for x, y in joint[:, :2]], shape=image.shape)
    data, kpt_aug = aug(image=image, keypoints=kpt)
    keypoints_aug = None
    for i, point in enumerate(kpt_aug):
        if i == 0:
            keypoints_aug = [[point.x, point.y]]
        else:
            keypoints_aug = np.append(keypoints_aug, [[point.x, point.y]], axis=0)
    kpt_all.append(keypoints_aug)
    imageio.imwrite('augmented/'+str(index)+'.jpg',data)
kpt_all=np.array(kpt_all)

mat = np.concatenate((kpt_all,joints[:,:,2,np.newaxis]), axis=2)
mat = np.transpose(mat,(2,1,0))
savemat('augmented/aug.mat',{'joints':mat})
