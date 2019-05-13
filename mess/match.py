"""match the jpg with annotation"""

import os
import glob
import shutil

test_dir = r'E:\data\rain_full\rain_imgs\test'
test_dir_list = glob.glob(os.path.join(test_dir,'*'))
annotation_dir = r'E:\data\rain_full\Annotation'
annotation_list = os.listdir(annotation_dir)

match = []
for test_path in test_dir_list[1:]:
    test_list = os.listdir(test_path)



    for jpg in test_list:
        jpg_xml = jpg[:-4] + '.xml'
        if jpg_xml in annotation_list:
            shutil.copy(os.path.join(test_path,jpg), r'E:\data\rain_full\rain_imgs\match\img')
            shutil.copy(os.path.join(annotation_dir, jpg_xml), r'E:\data\rain_full\rain_imgs\match\xml')
            match.append(jpg[:-4])
    pass


pass





