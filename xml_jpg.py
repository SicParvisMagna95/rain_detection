import os
import xml.dom.minidom
import cv2

ImgPath = r'E:\data\rain_full\rain_imgs\match\img'
AnnoPath = r'E:\data\rain_full\rain_imgs\match\xml'
ImgLabelPath = r'E:\data\rain_full\rain_imgs\match\img_label'

imagelist = os.listdir(ImgPath)
for image in imagelist:

    image_pre, ext = os.path.splitext(image)
    imgfile = os.path.join(ImgPath, image)
    xmlfile = os.path.join(AnnoPath, image_pre+'.xml')

    # 打开xml文档
    DOMTree = xml.dom.minidom.parse(xmlfile)
    # 得到文档元素对象
    collection = DOMTree.documentElement
    # 读取图片
    img = cv2.imread(imgfile)

    # filenamelist = collection.getElementsByTagName("filename")
    # filename = filenamelist[0].childNodes[0].data
    # print(filename)
    # 得到标签名为object的信息
    objectlist = collection.getElementsByTagName("object")

    for objects in objectlist:
        # 每个object中得到子标签名为name的信息
        # namelist = objects.getElementsByTagName('name')
        # # 通过此语句得到具体的某个name的值
        # objectname = namelist[0].childNodes[0].data

        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            # cv2.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
            #            thickness=2)
    # cv2.imshow('head', img)
    # cv2.waitKey()

    cv2.imwrite(os.path.join(ImgLabelPath, image), img)   #save picture
