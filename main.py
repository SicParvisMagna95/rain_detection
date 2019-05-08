import torch
import dataset
import model
import cv2
import os
import torch.nn as nn
import numpy as np
import time
import torch.cuda as cuda
import glob

if __name__ == '__main__':

    gpu_avail = cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    scale = 0.8
    img_dir = '/home/zhangtk/data/test/2/'
    # img_dir = r'E:\data\rain_full\rain_imgs\test\2'
    # img_list = os.listdir(img_dir)
    img_list_ = glob.glob(os.path.join(img_dir,'*.jpg'))
    img_list = [os.path.basename(i) for i in img_list_]

    # save_dir = f'/home/zhangtk/data/test/0_test_ztkvgg1_{scale}/'
    save_dir = os.path.join(img_dir, 'result_mobilenet_conv1x1_95')

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
            info = torch.load('./model_saved/mobilenet_conv1x1/accuracy_0.9507529149821292.pkl')
            img = img.to("cuda")
        else:
            info = torch.load('./model_saved/mobilenet_conv1x1/accuracy_0.9507529149821292.pkl', map_location='cpu')


        net = info['model']

        # net = net.module


        net = net.eval()
        # print(net)

        s = time.time()

        out = net(img)
        # cv2.imshow('i',out[0,:,:].detach().cpu().numpy())
        # cv2.waitKey()
        out = torch.unsqueeze(out, dim=0)
        out = nn.Softmax2d()(out)
        out = out[0,1,:,:]
        out = out.cpu().detach().numpy()
        out = cv2.resize(out,(w,h))
        out = out*255
        # out = out[:,:,np.newaxis]
        # cv2.imshow('abc',out)
        # cv2.waitKey(0)

        print(img_path)

        e = time.time()
        # print(e-s, ' s')

        cv2.imwrite(os.path.join(save_dir, img_path[:-4]+'_test.jpg'), out)
        # cv2.imshow('img', out)
        # cv2.waitKey()
        pass


    end = time.time()

    print(end-start, ' s')







