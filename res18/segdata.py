import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 3:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            label_name1 = os.path.join(data_root, line_split[2])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name, label_name1)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SegData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path, label_path1 = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # GRAY 1 channel ndarray with shape H * W
        if 'ignore' not in label_path:
            label[label==255]=1
        label1 = cv2.imread(label_path1, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if 'ignore' not in label_path1:
            label1[label1==255]=1
        if self.transform is not None:
            image, label, label1 = self.transform(image, label,label1)
        return image, label, label1