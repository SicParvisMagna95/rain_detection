import torch
import dataset
import model
import cv2
import os
import torch.nn as nn
import numpy as np
import time
import torch.cuda as cuda

if __name__ == '__main__':

    gpu_avail = cuda.is_available()

    scale = 0.8
    img_dir = r'E:\data\tag_rain\img_untagged\score_40000_50000\no_rain'
    img_list = os.listdir(img_dir)
    # save_dir = f'/home/zhangtk/data/test/0_test_ztkvgg1_{scale}/'
    save_dir = os.path.join(img_dir, 'result_ztkvgg')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


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

        if gpu_avail:
            info = torch.load('./model_saved/accuracy_0.9768754516513349.pkl')
            img = img.to("cuda")
        else:
            info = torch.load('./model_saved/accuracy_0.9768754516513349.pkl', map_location='cpu')


        net = info['model']

        net = net.module

        net = net.eval()
        # print(net)

        s = time.time()

        out = net(img)
        out = torch.unsqueeze(out, dim=0)
        out = nn.Softmax2d()(out)
        out = out[0,1,:,:]
        out = out.cpu().detach().numpy()
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







