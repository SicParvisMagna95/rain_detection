import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
import glob
import pickle
import numpy as np


class RainData(Data.Dataset):
    def __init__(self, train=True):
        super(RainData, self).__init__()

        root = '/home/chauncy/data/cropped'

        self.train_data = []

        if train:
            data_path = [os.path.join(root, 'train_True', '32.imdb'),
                              os.path.join(root, 'train_False', '32.imdb')]
        else:
            data_path = [os.path.join(root, 'val_True', '32.imdb'),
                              os.path.join(root, 'val_False', '32.imdb')]

        for file_train in data_path:
            imdb = open(file_train, 'rb')
            data = pickle.load(imdb)
            self.train_data += data
            imdb.close()
        pass

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        img = (self.train_data[item][0]).astype(np.float)
        label = float((self.train_data[item][1]))

        return img, label
