import matplotlib.pyplot as plt
import matplotlib
from util.utils import *
from util.metric import *

def matplotlib_imshow(img, one_channel=True):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + .5
    img_numpy = img.numpy()

    if one_channel:
        plt.imshow(img_numpy, cmap='Greys')
    else:
        plt.imshow(np.transpose(img_numpy, (1, 2, 0)))


def plot_classes_predictions(net, images, labels, classes):
    preds, probs = get_pred_prob(net, images)

    fig = plt.figure(figsize=(12, 48))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], True)
        ax.set_title('{0}, {1:.1f}%\nLabel:{2}'.format(classes[preds[idx]], probs[idx] * 100, labels[idx]),
                     color='green' if preds[idx] == labels[idx] else 'red')

    return fig


def plot_joints(net, images, bboxes, classes=None):
    all_joints = get_joints_coords(net, images.to('cuda'), bboxes)
    images = unnormalize_image(images)
    images = np.transpose(images, (0, 2, 3, 1))
    fig = plt.figure(figsize=(12, 48))
    for idx in range(np.minimum(4, images.shape[0])):
        joints = all_joints[idx]
        #joints = unnormalize_joints(joints, bboxes[idx])
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        plt.imshow(images[idx])
        limbs = joints_to_limbs(joints)
        for limb in limbs:
            plt.scatter(limb[1], limb[0], cmap='magma')
            plt.plot(limb[1], limb[0], linewidth=3)
        #plt.scatter(joints[:, 1], joints[:, 0])
        ax.set_title('images')
        #import imageio
        #imageio.imwrite('3.jpg',images[idx])
    return fig
