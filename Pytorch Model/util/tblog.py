from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import shutil


class TbLogger():
    def __init__(self, logdir, unique_dir=False, restart=False):
        if restart:
            subdirectories = [f.path for f in os.scandir(logdir) if f.is_dir()]
            for dir in subdirectories:
                shutil.rmtree(dir)
        if unique_dir:
            logdir = os.path.join(logdir, str(datetime.datetime.today().strftime('%d. %B %Y')))
            print(logdir)
        self.writer = SummaryWriter(logdir)

    def add_image(self, title, image, global_step=None):
        self.writer.add_image(title, image, global_step)

    def add_graph(self, net, image, verbose=False):
        self.writer.add_graph(net, image, verbose)

    def add_embedding(self, features, metadata, label_img, global_step=None):
        self.writer.add_embedding(features, metadata, label_img, global_step=global_step)

    def add_scalar(self, var, val, global_step=None):
        self.writer.add_scalar(var, val, global_step)

    def add_figure(self, var, image, global_step=None):
        self.writer.add_figure(var,image,global_step)

    def close(self):
        self.writer.close()
