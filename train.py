import model
import dataset
import torch
import torch.cuda as cuda
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as Optim
from tensorboardX import SummaryWriter

EPOCHS = 100
BATCH_SIZE = 512


if __name__ == '__main__':
    # gpu
    gpu_avail = cuda.is_available()

    net = model.my_vgg()
    if gpu_avail:
        net = net.to("cuda")

    input_train_data = dataset.RainData(train=True)
    input_test_data = dataset.RainData(train=False)

    train_data_loader = Data.DataLoader(input_train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_data_loader = Data.DataLoader(input_test_data,batch_size=BATCH_SIZE,shuffle=False)



    optimizer = Optim.Adam(net.parameters())

    loss_func = nn.CrossEntropyLoss()

    print(net)

    for epoch in range(EPOCHS):
        net.train()

        loss_train_epoch = 0.0
        loss_test_epoch = 0.0

        for i,(img, label) in enumerate(train_data_loader):

            # pytorch has to do these type cvt
            # img---float32(float), label---int64(long)
            img = img.float()
            label = label.long()

            if gpu_avail:
                img = img.to('cuda')
                label = label.to('cuda')

            with SummaryWriter(log_dir='./model_graph', comment='my_vgg') as w:
                w.add_graph(net, img)

            optimizer.zero_grad()

            out = net(img)                  # (Batch_size, 2)
                                            # value_range over channels (0.0, 1.0)

            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            loss_train_epoch += loss.item() * BATCH_SIZE

            print(f"Batch {i}\tTrain loss: {loss.item()}")


        # # evaluation
        # net.eval()
        #
        # for j, (img_val, label_val) in enumerate(test_data_loader):
        #
        #     img_val = img_val.float()
        #     label_val = label_val.long()
        #
        #     if gpu_avail:
        #         img_val = img_val.to("cuda")
        #         label_val = label_val.to("cuda")
        #     out_val = net(img_val)
        #     loss_val = loss_func(out_val, label_val)
        #     loss_test_epoch += loss_val.item() * BATCH_SIZE
        #
        # print(f'Train loss: {loss_train_epoch/train_data_loader.__len__()}\t'
        #       f'Val loss: {loss_test_epoch/test_data_loader.__len__()}')
        pass


















