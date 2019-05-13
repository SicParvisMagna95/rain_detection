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

        root = '/data/rain/rain_detection_train/cropped/'

        self.train_data = []

        if train:
            data_path = [os.path.join(root, 'train_True', '32.imdb'),
                              os.path.join(root, 'train_False', '32.imdb')]
        else:
            data_path = [os.path.join(root, 'val_True', '32.imdb'),
                              os.path.join(root, 'val_False', '32.imdb')]

        for file_train in data_path:
        # for j in range(2):
            imdb = open(file_train, 'rb')
            data = pickle.load(imdb)
            if 'True' in file_train:
                self.train_data += [[i,1] for i in data]
                # self.train_data += np.column_stack((data, np.ones(len(data),dtype=np.float)))
            else:
                self.train_data += [[i, 0] for i in data]
                # print(0)
                # self.train_data += np.column_stack((data, np.zeros(len(data),dtype=np.float)))
            imdb.close()
        pass

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        img = (self.train_data[item][0]).astype(np.float)
        label = float((self.train_data[item][1]))

        return img, label

