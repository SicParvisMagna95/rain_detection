import model
import dataset
import torch
import torch.cuda as cuda
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as Optim
import matplotlib.pyplot as plt
import os
from visdom import Visdom
import numpy as np
from loss import FocalLoss,FocalLoss_bce



# hyper-parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
EPOCHS = 40
BATCH_SIZE = 500


if __name__ == '__main__':
    # gpu
    gpu_avail = cuda.is_available()

    # model
    # net = model.Ztk_vgg()
    net = model.MobileNet()

    if cuda.device_count()>1:
        print('cuda count:',cuda.device_count())
        net = nn.DataParallel(net)

    # names = net.named_parameters()
    # for name,p in names:
    #     print(name)
    #     pass

    if gpu_avail:
        net = net.to("cuda")

    # make directory
    if not os.path.exists('./loss'):
        os.makedirs('./loss')

    model_saved_path = './model_saved_new_3/mobilenet_conv1x1'
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    # data
    input_train_data = dataset.RainData(train=True)
    input_test_data = dataset.RainData(train=False)


    train_data_loader = Data.DataLoader(input_train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_data_loader = Data.DataLoader(input_test_data,batch_size=BATCH_SIZE,shuffle=False)

    # optimizer
    optimizer = Optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=1000)


    # loss function
    loss_func = nn.CrossEntropyLoss()
    # loss_func = FocalLoss(2,alpha=0.25)
    # loss_func = FocalLoss_bce(alpha=1, gamma=2)

    # visualize
    env = Visdom(port=10022,env='mobilenet')

    # initialize some variables
    accuracy,accuracy_old = 0,0
    train_loss_curve,val_loss_curve = [],[]

    iteration = 0
    # start training
    for epoch in range(EPOCHS):


        # loss_train_epoch = 0.0
        # loss_test_epoch = 0.0
        net.train()

        for i,(img, label) in enumerate(train_data_loader):

            net.train()

            iteration = int(epoch*(len(input_train_data.train_data)/BATCH_SIZE+1)+i+1)
            # pytorch has to do these type cvt
            # img---float32(float), label---int64(long)
            img = img.float()
            label = label.long()

            if gpu_avail:
                img = img.to('cuda')
                label = label.to('cuda')

            # with SummaryWriter(log_dir='./model_graph', comment='my_vgg') as w:
            #     w.add_graph(net, img)

            optimizer.zero_grad()

            out = net(img)                  # (Batch_size, 2)
                                            # value_range over channels (0.0, 1.0)

            loss = loss_func(out, label)


            loss.backward()
            # optimizer.step()
            scheduler.step(loss)

            # loss_train_epoch += loss.item() * label.shape[0]
            loss_train_iteration = loss.item()

            if i % 200 == 0:
                # print(f"Epoch {epoch}\tIteration {i}\tTrain loss: {loss.item()}")

                # env.text(f"Epoch {epoch}\tIteration {i}\tTrain loss: {loss.item()}\n",
                #      win='log',env='mobilenet',
                #      append=False if iteration == 1 else True
                #      )

                # print(epoch, iteration)
                # env.line(Y=np.array([loss.item()]),X=np.array([iteration]),
                #          win='loss',env='mobilenet',
                #          opts=dict(title='loss',
                #                    xlabel='iteration',ylabel='loss',
                #                    legend=['Train loss'],
                #                    showlegend=True
                #                    ),
                #          update=None if iteration == 1 else 'append')


                # evaluation
                net.eval()

                correct = 0
                loss_test_iteration = 0.0
                for j, (img_val, label_val) in enumerate(test_data_loader):

                    img_val = img_val.float()
                    label_val = label_val.long()

                    if gpu_avail:
                        img_val = img_val.to("cuda")
                        label_val = label_val.to("cuda")
                    out_val = net(img_val)      # (Batch_size, 2)

                    # output the accuracy
                    out_class = torch.argmax(out_val, dim=1)
                    count = torch.sum(out_class==label_val).item()
                    # print(count)
                    correct += count

                    loss_val = loss_func(out_val, label_val)
                    # loss_test_epoch += loss_val.item() * label_val.shape[0]
                    loss_test_iteration += loss_val.item() * label_val.shape[0]

                # loss_total_train = loss_train_iteration/len(input_train_data.train_data)
                # loss_total_train = loss_train_iteration
                loss_total_val = loss_test_iteration/len(input_test_data.train_data)

                accuracy = correct/len(input_test_data.train_data)


                if accuracy_old<accuracy:
                    accuracy_old = accuracy


                    torch.save({'model':net,
                                'epoch':epoch,
                                'batch_size':BATCH_SIZE,
                                },f'{model_saved_path}/accuracy_{accuracy}.pkl')

                # record train_loss,val_loss
                # train_loss_curve.append(loss_total_train)
                train_loss_curve.append(loss_train_iteration)
                val_loss_curve.append(loss_total_val)

                env.line(Y=np.column_stack((loss_train_iteration,loss_total_val)),
                         # X=np.column_stack((epoch,epoch)),
                         X=np.array([iteration]),
                         win='loss', env='mobilenet',
                         opts=dict(title='Loss',
                                   xlabel='Iteration', ylabel='loss',
                                   legend=['Train loss','Validation loss'],
                                   markers=True,
                                   showlegend=True
                                   ),
                         update=None if iteration == 201 else 'append')

                env.line(Y=np.array([accuracy]),
                         # X=np.column_stack((epoch,epoch)),
                         X=np.array([iteration]),
                         win='accuracy', env='mobilenet',
                         opts=dict(title='Accuracy',
                                   xlabel='Iteration', ylabel='accuracy',
                                   legend=['Accuracy'],
                                   markers=True,
                                   showlegend=True
                                   ),
                         update=None if iteration == 1 else 'append')

                print(f'Iteration {iteration} finished! Train loss: {loss_train_iteration}\t'
                      f'Val loss: {loss_total_val}\t'
                      f'Accuracy: {accuracy}')

                env.text(f'Iteration {iteration} finished! Train loss: {loss_train_iteration}\t'
                         f'Val loss: {loss_total_val}\t'
                         f'Accuracy: {accuracy}\n\n',
                         win='log', env='mobilenet',
                         append=False if iteration == 1 else True)

                print()
                pass


        print(f'Epoch {epoch} finished!\n')
        env.text(f'Epoch {epoch} finished!\n',
                 win='log', env='mobilenet',
                 append=False if iteration == 1 else True)

    plt.plot(range(1,EPOCHS+1), train_loss_curve, label='train loss')
    plt.plot(range(1,EPOCHS+1), val_loss_curve, label='val loss')
    plt.title('Mobilenet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'./loss/loss_epoch{EPOCHS}')
    plt.show()
