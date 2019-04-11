import torch
import dataset
import model
import cv2
import os
import torch.nn as nn
import numpy as np
import time




if __name__ == '__main__':

    scale = 1.2
    img_dir = r'E:\data\rain_full\rain_imgs\match\img'
    img_list = os.listdir(img_dir)
    save_dir = f'E:/data/rain_full/rain_imgs/match/all_{scale}'

    start = time.time()

    for img_path in img_list:
        img = cv2.imread(os.path.join(img_dir, img_path))

        h = img.shape[0]
        w = img.shape[1]

        img = cv2.resize(img,(0,0),fx=scale,fy=scale)
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img,dim=0)
        img = img.permute(0,3,1,2)
        info = torch.load('./model_saved/accuracy_0.9803812425538564.pkl', map_location='cpu')

        net = info['model']

        net = net.eval()
        # print(net)

        s = time.time()

        out = net(img)
        out = torch.unsqueeze(out, dim=0)
        out = nn.Softmax2d()(out)
        out = out[0,1,:,:]
        out = out.detach().numpy()
        out = cv2.resize(out,(w,h))
        out = out*255
        # out = out[:,:,np.newaxis]

        e = time.time()
        # print(e-s, ' s')

        cv2.imwrite(os.path.join(save_dir, img_path[:-4]+'_test.jpg'), out)
        # cv2.imshow('img', out)
        # cv2.waitKey()
        pass


    end = time.time()

    print(end-start, ' s')







