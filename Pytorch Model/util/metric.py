import numpy as np
from util.utils import *


def joints_to_limbs(joints):
    limbs_order = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [12, 13]]
    limbs = [[[joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]]] for i, j in limbs_order]
    return limbs


def pcp_metric(real_joints, predicted_joints, threshold=0.5):
    real_limbs = np.array(joints_to_limbs(real_joints))
    predicted_limbs = np.array(joints_to_limbs(predicted_joints))
    n_limbs = real_limbs.shape[0]
    running_mean = 0
    limb_distances=[]
    for real_limb, predicted_limb in zip(real_limbs, predicted_limbs):
        length = np.linalg.norm(
            np.array([real_limb[0, 0], real_limb[1, 0]]) - np.array([real_limb[0, 1], real_limb[1, 1]]))
        dist_1 = np.linalg.norm(np.array([real_limb[0, 0], real_limb[1, 0]]) - np.array(
            [predicted_limb[0, 0], predicted_limb[1, 0]])) / length
        dist_2 = np.linalg.norm(np.array([real_limb[0, 1], real_limb[1, 1]]) - np.array(
            [predicted_limb[0, 1], predicted_limb[1, 1]])) / length
        mean_dist = (dist_1 + dist_2) / 2
        # if mean_dist < threshold:
        #     running_mean += mean_dist
        limb_distances.append(mean_dist)
    limb_distances=np.array(limb_distances)
    return limb_distances


def pdj_metric(real_joints, predicted_joints, length=10):
    real_limbs = np.array(joints_to_limbs(real_joints))
    predicted_limbs = np.array(joints_to_limbs(predicted_joints))
    n_limbs = real_limbs.shape[0]
    distances =[]
    for real_joint, predicted_joint in zip(real_joints, predicted_joints):
        dist_1 = np.linalg.norm(real_joint- predicted_joint)
        distances.append(dist_1)

    distances = np.array(distances)
    return distances


def metrics(net, images, bboxes, real_joints, metric_type='pcp'):
    predicted_joints = get_joints_coords(net, images, bboxes)
    predicted_joints = predicted_joints[0]
    real_joints = real_joints.numpy()
    real_joints = real_joints[0]
    predicted_joints_ordered = []
    real_joints_ordered = []
    # for j in range(int(predicted_joints.shape[0] / 2)):
    #     predicted_joints_ordered.append([predicted_joints[2 * j], predicted_joints[2 * j + 1]])
    for j in range(int(real_joints.shape[0] / 2)):
        real_joints_ordered.append([real_joints[2 * j], real_joints[2 * j + 1]])

    real_joints_ordered = np.array(real_joints_ordered)
    real_joints_ordered = unnormalize_joints(real_joints_ordered, bboxes[0])
    predicted_joints_ordered = np.array(predicted_joints_ordered)
    if metric_type == 'pcp':
        metric_value = pcp_metric(real_joints_ordered, predicted_joints, 0.5)
    elif metric_type=='pdj':
        metric_value = pdj_metric(real_joints_ordered, predicted_joints, 20)
    else:
        raise ValueError('The metric type {} is invalid'.format(metric_type))

    return metric_value
