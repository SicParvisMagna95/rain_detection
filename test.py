import cv2
import torch
import time
import torch.cuda as cuda


gpu_avail = cuda.is_available()
if gpu_avail:
    info = torch.load('./model_saved/accuracy_0.9803812425538564.pkl')
else:
    info = torch.load('./model_saved/accuracy_0.9803812425538564.pkl', map_location='cpu')

net = info['model']

number = 10

print('\nTest on cpu:')
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
