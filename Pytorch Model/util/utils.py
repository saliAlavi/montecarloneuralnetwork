import torch
import torch.nn.functional as F
import numpy as np
import os
import re
from pathlib import Path


def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    index = torch.randperm(len(data))
    return data[index][:n], labels[index][:n]


def get_pred_prob(net, images):
    outputs = net(images)
    _, preds_tensor = torch.max(outputs, dim=1)
    preds = np.squeeze(preds_tensor.numpy())
    probs = [F.softmax(el, dim=0)[i] for i, el in zip(preds, outputs)]
    return preds, probs


def get_joints_pred(net, images):
    outputs = net(images.to('cuda:0'))
    joints = []
    outputs = outputs.to('cpu')
    outputs = outputs.detach().numpy()
    all_joints = []
    for i in range(outputs.shape[0]):
        joints = []
        for j in range(int(outputs.shape[1] / 2)):
            joints.append([outputs[i, 2 * j], outputs[i, 2 * j + 1]])
        all_joints.append(joints)
    all_joints = np.array(all_joints)
    return all_joints


def get_joints_coords(net, images,bboxes):
    joints_normal = get_joints_pred(net, images)
    joints = [unnormalize_joints(joints, bboxes[idx]) for idx, joints in enumerate(joints_normal)]
    return joints


def save_model(model, optimizer, epoch, iteration, path):
    files_list = Path(os.path.join('data', 'model')).glob('*.pt')
    files_list = [re.sub('\.pt$', '', os.path.split(f)[1]) for f in list(files_list)]
    if files_list:
        last_file = (int(max(files_list)))
    else:
        last_file = -1
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }, os.path.join(path, '{0:04d}.pt'.format(last_file + 1)))


def load_model(model, optimizer, path, latest=True):
    if latest:
        files_list = Path(os.path.join('data', 'model')).glob('*.pt')
        files_list = [re.sub('\.pt$', '', os.path.split(f)[1]) for f in list(files_list)]
        last_file = (int(max(files_list)))
        path = os.path.join(path, '{0:04d}.pt'.format(last_file))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']

    return model, optimizer, epoch, iteration


def unnormalize_joints(joints, target_shape):
    x,y, h, w = target_shape
    joints_arranged = []
    joints = list(joints)
    # joints_t = [[joint[0], joint[1]] for joint in joints]
    # joints=joints_t
    joints = np.array(joints)
    joints = np.array(list(joints))
    unnorm_joints = [[x, y] for x, y in zip(h / 2 * joints[:, 0] + h / 2, w / 2 * joints[:, 1] + w / 2)]
    unnorm_joints = np.array(unnorm_joints)
    unnorm_joints = np.clip(unnorm_joints, 0, np.inf)
    return unnorm_joints


def unnormalize_image(image):
    image = image * 128 + 128
    image = np.clip(image.numpy(), 0, 255).astype(int)
    return image

def moving_average(vector, n):
    vector_cumsum = np.cumsum(vector,dtype=float)
    vector_cumsum[n:] = vector_cumsum[n:] - vector_cumsum[:-n]
    return vector_cumsum[n-1:]/n