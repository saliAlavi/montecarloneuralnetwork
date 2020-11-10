from util.dataset import *
from util.loss import *
from util.optimizer import *
from util.plot import *
from util.tblog import *
from util.metric import *
from options.trainoptions import *
from model.nets import *
import tensorflow as tf
import tensorboard as tb
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

if __name__ == '__main__':
    option = TrainOptions()
    args = option.getArgs()

    if args.device == '':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    train_test_split=0.3
    dataset_size = len(PolDataset('dataset'))
    indices=list(range(dataset_size))
    split=int(np.floor(dataset_size*train_test_split))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(PolDataset('dataset'), batch_size=args.batchsize,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(PolDataset('dataset'), batch_size=args.batchsize,
                                                    sampler=test_sampler )

    dl = getDataloader(dataset=args.dataset, savedir=args.traindsdir, batchsize=args.batchsize, train=True)
    net = LinearNet(250)
    net.to(device)
    criterion = l2norm
    optimizer = getOptimizer(net.parameters(), 'adam', {'lr': args.lr, 'momentum': args.momentum})
    if args.lrdecay:
        scheduler = getScheduler(optimizer, sch_type=args.lrdecaytype)
    if args.log:
        writer = TbLogger(args.logdir, unique_dir=True, restart=True)


    running_loss = 0.0
    running_acc = 0.0
    running_loss_test=0.0
    running_acc_all = []
    running_loss_all = []

    start_epoch = 0
    if args.resume:
        if args.resumefrom == -1:
            net, optimizer, start_epoch, start_iteration = load_model(net, optimizer, args.loadpath, latest=True)
        else:
            net, optimizer, start_epoch, start_iteration = load_model(net, optimizer, os.path.join(args.loadpath,
                                                                                                   '{0:04d}.pt'.format(
                                                                                                       args.resumefrom)),
                                                                      latest=False)
    for epoch in range(start_epoch, args.epochs):
        for i, data in enumerate(train_loader, 0):
            cat_data,n = data
            net.train()
            optimizer.zero_grad()
            outputs = net(cat_data.to(device))
            loss = criterion(outputs, n.to(device) )
            loss.backward()
            optimizer.step()
            if args.lrdecay:
                scheduler.step(loss)
            print(loss.item())
            # with torch.no_grad():
            #     net.eval()
            #     for j, data_test in enumerate(test_loader):
            #         images_test, points_test, valids_test, bboxes_test = data_test
            #         outputs_test = net(co_data.to(device))
            #         loss_test = criterion(outputs_test, points_test.to(device), valids_test.to(device))
            #         running_loss_test+=loss_test.item()
            #         break
            #     running_loss += loss.item()
            #     # running_acc_all.append(acc)
            #     running_loss_all.append(loss.item())
                # if args.save:
                #     if (len(dl) * epoch + i) % args.saveevery == args.saveevery - 1:
                #         save_model(net, optimizer, epoch, i, os.path.join('data', 'model'))
                #         print(
                #             'Saving the model in Epoch: {0}, Iteration: {1}, , Accuracy: {2:.4f}%, Loss: {3:.6f}'.format(
                #                 epoch + 1, i + 1,
                #                 np.sum(np.asarray(running_acc_all[-args.printevery:])) / args.printevery * 100,
                #                 np.sum(np.asarray(running_loss_all[-args.printevery:])) / args.printevery))
                #
                # if args.verbose:
                #     if i % args.printevery == args.printevery - 1:
                #         print('Epoch: {0}, Iteration: {1}, Accuracy: {2:.4f}%, Loss: {3:.6f}'.format(
                #             epoch + 1, i + 1,
                #             np.sum(np.asarray(running_acc_all[-args.printevery:])) / args.printevery * 100,
                #             np.sum(np.asarray(running_loss_all[-args.printevery:])) / args.printevery))
                #
                # if args.log:
                #     if i % args.tbsaveevery == args.tbsaveevery - 1:
                #
                #             for i in range(11):
                #                 writer.add_scalar('PCP {}'.format(limbs_name[i]), pcp_metrics_current[i], global_step=epoch * len(dl) + i + 1)
                #             for i in range(14):
                #                 writer.add_scalar('PDJ {}'.format(joints_name[i]), pdj_metrics_current[i], global_step=epoch * len(dl) + i + 1)
                #
                #         writer.add_figure('prediction',
                #                           plot_joints(net, images, bboxes), global_step=epoch * len(dl) + i+1)
                #         writer.add_scalar('Train Loss', running_loss/args.tbsaveevery, global_step=epoch * len(dl) + i+1)
                #         writer.add_scalar('Test Loss', running_loss_test/split, global_step=epoch * len(dl) + i + 1)
                #         running_loss = 0.0
                #         running_loss_test=0.0
                #         running_acc = 0.0
    if args.log:
        writer.close()
    print('Training Finished!')
