import argparse

class parse():
    def __init__(self):
        self.parser  = argparse.ArgumentParser()
        self.parser.add_argument('--nclasses', help='Number of classes', type= int, default=10)

    def getArgs(self):
        self.args = self.parser.parse_args()
        return self.args
