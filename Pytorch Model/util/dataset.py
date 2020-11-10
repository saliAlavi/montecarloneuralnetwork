from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
from scipy.io import loadmat
import numpy as np
import torch
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from skimage import io
import itertools
import torchvision.transforms.functional as F
from PIL import Image
import copy
import cv2 as cv
import matplotlib.pyplot as plt
import imageio
from util.dataaugment import *
import csv


class PolDataset(Dataset):
    def __init__(self, root='data'):
        super(PolDataset, self).__init__()
        self.root = root
        self.zstokes = list(sorted(os.listdir(os.path.join(root, 'zstokes'))))

        self.ns=[]
        with open(os.path.join(root, 'params.txt')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.ns.append(row[1])

    def __len__(self):
        return len(self.zstokes)

    def __getitem__(self, item):
        zstokes = np.load(os.path.join(self.root,'zstokes',self.zstokes[item]))[:,0]

        n = float(self.ns[item])
        n=torch.tensor([n])
        trfm=transforms.Compose([transforms.ToTensor()])

        zstokes=torch.from_numpy(zstokes)
        # zstokes = (zstokes-torch.mean(zstokes))/torch.std(zstokes)

        zstokes=torch.squeeze(zstokes)

        return zstokes  ,n




def getDataloader(dataset='pol', transform=None, savedir='./datasets', train=True, batchsize=4):

    shuffle = True if train else False
    if dataset == 'pol':
        data = PolDataset('dataset')
    else:
        raise ValueError('Invalid dataset root directory: {}'.format(dataset))

    torch.multiprocessing.freeze_support()
    dataLoader = torch.utils.data.DataLoader(data,
                                             batch_size=batchsize,
                                             shuffle=shuffle,
                                             num_workers=1)
                                             # collate_fn=PadCollate()
                                             # )

    return dataLoader


def getDataset(dataset='pol', transform=None, train=True):
    if transform == None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    shuffle = True if train else False
    if dataset == 'pol':
        data = PolDataset('dataset')
    else:
        raise ValueError('Invalid dataset root directory: {}'.format(dataset))

    return data


# class PadCollate:
#     def __init__(self):
#         pass
#
#     def __call__(self, batch):
#         images, points, valids = zip(*batch)
#
#         images = [np.array(image) for image in images]
#
#         original_shape = images[0].shape
#         images_bboxes = [list(self.normalize_image(image, point_single)) for image, point_single in zip(images, points)]
#         images_bboxes = np.array(images_bboxes)
#         bboxes = list(images_bboxes[:, 0])
#         images = list(images_bboxes[:, 1])
#
#         original_shape = images[0].shape
#         # imageio.imwrite(str(0) + '.jpg', images[0])
#         max_length = np.max(list(map(lambda x: [x.shape[0], x.shape[1]], images)), axis=0)
#         images, bboxes = self.resize_images(images, max_length, bboxes)
#         images = [self.pad_images(image, max_length[0], 0) for image in images]
#         images = [self.pad_images(image, max_length[1], 1) for image in images]
#         points = [[self.normalize_points(point_single, bbox)] for point_single, bbox in zip(points, bboxes)]
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((128, 128, 128), (128, 128, 128))])
#         # imageio.imwrite(str(1) + '.jpg', images[0])
#         images = [transform(image) for image in images]
#         images = [torch.clamp(image, -1, 1) for image in images]
#         points = [torch.tensor(points_single, dtype=torch.float64) for points_single in points]
#         points = [torch.flatten(points_single) for points_single in points]
#         valids = [torch.tensor(valid) for valid in valids]
#         images = torch.stack(images, dim=0).float()
#         points = torch.stack(points, dim=0).float()
#         valids = torch.stack(valids, dim=0).float()
#         return images, points, valids, bboxes
#
#     def pad_images(self, vec, max_l, dim):
#         pad_size = list(vec.shape)
#         pad_size[dim] = max_l - vec.shape[dim]
#         vec_cat = np.concatenate((vec, np.zeros(tuple(pad_size))), axis=dim)
#         return vec_cat
#
#     def normalize_image(self, image, points):
#         point_min = np.min(points, axis=0)
#         point_max = np.max(points, axis=0)
#         x_min = int(np.floor(point_min[0]))
#         y_min = int(np.floor(point_min[1]))
#         x_max = int(np.floor(point_max[0]))
#         y_max = int(np.floor(point_max[1]))
#         x_min = np.maximum(0, x_min)
#         y_min = np.maximum(0, y_min)
#         image = image[x_min:x_max, y_min:y_max, :]
#         image = np.array(image)
#         bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
#         return bbox, image
#
#     def normalize_points(self, points, bbox):
#         x, y, h, w = bbox
#         points = [[((point[0] - x) / h - 0.5) / 0.5, ((point[1] - y) / w - 0.5) / 0.5] for point in points]
#
#         return points
#
#     def resize_images(self, images, max_length, bboxes):
#         images_resized = []
#         bboxes_resize = []
#         for idx, image in enumerate(images):
#
#             scale = max_length[0] / image.shape[0] if max_length[0] / image.shape[0] < max_length[1] / image.shape[
#                 1] else max_length[1] / image.shape[1]
#             images_resized.append(
#                 cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), None, cv.INTER_CUBIC))
#             x, y, h, w = bboxes[idx]
#             bboxes_resize.append((int(x * scale), int(y * scale), int(h * scale), int(w * scale)))
#         # images_resized = np.array(images_resized)
#         bboxes_resize = np.array(bboxes_resize)
#         return images_resized, bboxes_resize
#
#
#
#
# class PolDataset(Dataset):
#     def __init__(self, root='dataset'):
#         super(PolDataset, self).__init__()
#         self.root = root
#         self.co_paths = list(sorted(os.listdir(os.path.join(root, 'co'))))
#         self.cross_paths = list(sorted(os.listdir(os.path.join(root, 'cross'))))
#         self.incoh_paths = list(sorted(os.listdir(os.path.join(root, 'incoh'))))
#         self.nes=[]
#         with open(os.path.join(root, 'params.txt')) as csv_file:
#             csv_reader = csv.reader(csv_file, delimiter=',')
#             for row in csv_reader:
#                 self.nes.append(row[1])
#
#     def __len__(self):
#         return len(self.co_paths)
#
#     def __getitem__(self, item):
#         co_data = np.load(os.path.join(self.root,'co',self.co_paths[item]))
#         cross_data = np.load(os.path.join(self.root, 'cross',self.cross_paths[item]))
#         incoh_data = np.load(os.path.join(self.root, 'incoh',self.incoh_paths[item]))
#
#         co_data=np.mean(co_data,axis=0)
#         cross_data = np.mean(cross_data, axis=0)
#         incoh_data = np.mean(incoh_data, axis=0)
#
#         # co_data=torch.tensor(co_data)
#         # cross_data = torch.tensor(cross_data)
#         # incoh_data = torch.tensor(incoh_data)
#
#         n_e = float(self.nes[item])*100
#         n_e=torch.tensor([n_e])
#         trfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0 ],[0.5 ])])
#
#         co_data=trfm(co_data)
#         cross_data = trfm(cross_data)
#         incoh_data = trfm(incoh_data)
#
#         co_data=torch.squeeze(co_data)
#         cross_data = torch.squeeze(cross_data)
#         incoh_data = torch.squeeze(incoh_data)
#         return torch.stack([co_data,cross_data,incoh_data])  ,n_e
#
#
#
#
# def getDataloader(dataset='pol', transform=None, savedir='./datasets', train=True, batchsize=4):
#
#     shuffle = True if train else False
#     if dataset == 'pol':
#         data = PolDataset('dataset')
#     else:
#         raise ValueError('Invalid dataset root directory: {}'.format(dataset))
#
#     torch.multiprocessing.freeze_support()
#     dataLoader = torch.utils.data.DataLoader(data,
#                                              batch_size=batchsize,
#                                              shuffle=shuffle,
#                                              num_workers=1)
#                                              # collate_fn=PadCollate()
#                                              # )
#
#     return dataLoader
#
#
# def getDataset(dataset='pol', transform=None, train=True):
#     if transform == None:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#     shuffle = True if train else False
#     if dataset == 'pol':
#         data = PolDataset('dataset')
#     else:
#         raise ValueError('Invalid dataset root directory: {}'.format(dataset))
#
#     return data
#
#
# class PadCollate:
#     def __init__(self):
#         pass
#
#     def __call__(self, batch):
#         images, points, valids = zip(*batch)
#
#         images = [np.array(image) for image in images]
#
#         original_shape = images[0].shape
#         images_bboxes = [list(self.normalize_image(image, point_single)) for image, point_single in zip(images, points)]
#         images_bboxes = np.array(images_bboxes)
#         bboxes = list(images_bboxes[:, 0])
#         images = list(images_bboxes[:, 1])
#
#         original_shape = images[0].shape
#         # imageio.imwrite(str(0) + '.jpg', images[0])
#         max_length = np.max(list(map(lambda x: [x.shape[0], x.shape[1]], images)), axis=0)
#         images, bboxes = self.resize_images(images, max_length, bboxes)
#         images = [self.pad_images(image, max_length[0], 0) for image in images]
#         images = [self.pad_images(image, max_length[1], 1) for image in images]
#         points = [[self.normalize_points(point_single, bbox)] for point_single, bbox in zip(points, bboxes)]
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((128, 128, 128), (128, 128, 128))])
#         # imageio.imwrite(str(1) + '.jpg', images[0])
#         images = [transform(image) for image in images]
#         images = [torch.clamp(image, -1, 1) for image in images]
#         points = [torch.tensor(points_single, dtype=torch.float64) for points_single in points]
#         points = [torch.flatten(points_single) for points_single in points]
#         valids = [torch.tensor(valid) for valid in valids]
#         images = torch.stack(images, dim=0).float()
#         points = torch.stack(points, dim=0).float()
#         valids = torch.stack(valids, dim=0).float()
#         return images, points, valids, bboxes
#
#     def pad_images(self, vec, max_l, dim):
#         pad_size = list(vec.shape)
#         pad_size[dim] = max_l - vec.shape[dim]
#         vec_cat = np.concatenate((vec, np.zeros(tuple(pad_size))), axis=dim)
#         return vec_cat
#
#     def normalize_image(self, image, points):
#         point_min = np.min(points, axis=0)
#         point_max = np.max(points, axis=0)
#         x_min = int(np.floor(point_min[0]))
#         y_min = int(np.floor(point_min[1]))
#         x_max = int(np.floor(point_max[0]))
#         y_max = int(np.floor(point_max[1]))
#         x_min = np.maximum(0, x_min)
#         y_min = np.maximum(0, y_min)
#         image = image[x_min:x_max, y_min:y_max, :]
#         image = np.array(image)
#         bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
#         return bbox, image
#
#     def normalize_points(self, points, bbox):
#         x, y, h, w = bbox
#         points = [[((point[0] - x) / h - 0.5) / 0.5, ((point[1] - y) / w - 0.5) / 0.5] for point in points]
#
#         return points
#
#     def resize_images(self, images, max_length, bboxes):
#         images_resized = []
#         bboxes_resize = []
#         for idx, image in enumerate(images):
#
#             scale = max_length[0] / image.shape[0] if max_length[0] / image.shape[0] < max_length[1] / image.shape[
#                 1] else max_length[1] / image.shape[1]
#             images_resized.append(
#                 cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), None, cv.INTER_CUBIC))
#             x, y, h, w = bboxes[idx]
#             bboxes_resize.append((int(x * scale), int(y * scale), int(h * scale), int(w * scale)))
#         # images_resized = np.array(images_resized)
#         bboxes_resize = np.array(bboxes_resize)
#         return images_resized, bboxes_resize
