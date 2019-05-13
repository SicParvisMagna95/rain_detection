import cv2
import torch
import time
import torch.cuda as cuda
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_avail = cuda.is_available()
if gpu_avail:
    info = torch.load('./model_saved/accuracy_0.9762699947266655.pkl')
    print('\nTest on gpu:')
else:
    info = torch.load('./model_saved/accuracy_0.9762699947266655.pkl', map_location='cpu')
    print('\nTest on cpu:')
net = info['model']

number = 100


start = time.time()

img = cv2.imread('./RECORD2.20000201000408Front.mp4.432.jpg')
img = torch.from_numpy(img)
img = img.float()
img = torch.unsqueeze(img,dim=0)
img = img.permute(0,3,1,2)

if gpu_avail:
    img = img.to('cuda')

for i in range(number):
    a = net(img)

end = time.time()

print(end-start,f's for {number} images')





