import torch

#
# def l2normPointsLoss(outputs=torch.tensor([]), points=torch.tensor([]), valid=torch.tensor([])):
#     distance = torch.sub(outputs, points)
#     distance = distance.pow(2)
#     number_valid_joints = torch.sum(valid,dim = 1) / 2
#     temp = torch.sum(distance * valid, dim=1)
#     distance = torch.sum(distance * valid, dim=1) / number_valid_joints
#     distance = torch.sqrt(distance)
#     d=distance.size(0)
#     loss = torch.sum(distance)/distance.size(0)
#
#     return loss
#
#
# def getLoss(lossType='l2norm', params=None):
#     if lossType == 'l2norm':
#         loss = l2normPointsLoss
#     else:
#         raise ValueError('The loss type {} is not valid.'.format(lossType))
#
#     return loss


def l2norm(pred, target):
    norm = torch.sub(pred,target)
    norm = norm.pow(2)

    norm=torch.mean(norm)

    return norm