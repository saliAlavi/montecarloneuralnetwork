from options.mainParser import *
import os


class TrainOptions(parse):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=10 )
        self.parser.add_argument('--batchsize', help='Number of elements in each batch', type=int, default=8)
        self.parser.add_argument('--verbose', help='Print training results', action='store_true')
        self.parser.add_argument('--printevery', help='Print each number of iteration', type=int, default=1000)
        self.parser.add_argument('--tbsaveevery', help='Save loss to tensorboard every number of iterations', type=int,
                                 default=100)
        self.parser.add_argument('--traindsdir', help='Train dataset save directory', type=str, default='./datasets')
        self.parser.add_argument('--dataset', help='Dataset name', type=str, default='pol')
        self.parser.add_argument('--loss', help='Loss type', type=str, default='l2norm')
        self.parser.add_argument('--optim', help='Optimizer type', type=str, default='sgd')
        self.parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
        self.parser.add_argument('--momentum', help='Momentum for optimizer', type=float, default=0.9)
        self.parser.add_argument('--lrdecay', help = 'If reduce learning rate during training', action='store_true')
        self.parser.add_argument('--lrdecaytype', help='If lrdecay is set determines the decay type', type=str, default='plateau')
        self.parser.add_argument('--logdir', help='Tensorboard log directory', type=str, default='logs')
        self.parser.add_argument('--log', help='Log the training in tensorboard format', action='store_true')
        self.parser.add_argument('--save', help='Option to save model and optimizer', action='store_true')
        self.parser.add_argument('--saveevery', help='Save the model each number of iterations', type=int, default=500)
        self.parser.add_argument('--device', help='The device to run the model on [cuda:0, cpu]', type=str, default='')
        self.parser.add_argument('--resume', help='Resume training from last checkpoint', action='store_true')
        self.parser.add_argument('--resumerfrom', help='Resume training from this file number', type=int, default=-1)
        self.parser.add_argument('--loadpath',
                                 help='The directory/file path to load model from [without/with resumerfrom specified]',
                                 type=str, default=os.path.join('data', 'model'))
        self.parser.add_argument('--savepath', help='The path to save the model to', type=str,
                                 default=os.path.join('data', 'model'))
