import torch.optim as optim
import torch

def getOptimizer(netparams, optim_type='sgd', params=None):
    if optim_type == 'sgd':
        opt = optim.SGD(netparams, params['lr'], params['momentum'])
    elif optim_type == 'adam':
        opt = optim.Adam(netparams, params['lr'])
    elif optim_type=='sgd':
        opt = optim.SGD(netparams, params['lr'])
    else:
        raise ValueError('The optimzer type {} is not valid.'.format(optim_type))
    return opt

def getScheduler(optimizer, sch_type='', params=None):
    if sch_type == '':
        scheduler = None
    elif sch_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=500,verbose=True, )
    else:
        raise ValueError('Unknown scheduler type: {}'.format(sch_type))

    return scheduler