
import dataset
import model







if __name__ == '__main__':
    net = model.vgg16_bn(pretrained=True, num_classes=2)
    print(net)
    input_data = dataset.RainData(train=False)
    out = net(input_data)


    pass








