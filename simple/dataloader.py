import torch.utils.data as data
import os
import numpy as np
import cv2
#/mnt/lustre/share/dingmingyu/new_list_lane.txt

class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, new_width, new_height, label_width, label_height):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().split()
            imgs.append((words[0], words[1], words[2]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.height = new_height
        self.width = new_width
        self.label_height = label_height
        self.label_width = label_width
        print(imgs[0])

    def __getitem__(self, index):
        path, label1, label2 = self.imgs[index]
        path = os.path.join(self.dir_path, path)
        img = cv2.imread(path).astype(np.float32)
        img = img[:,:,:3]
        img -= [104, 117, 123]
        
        h, w = img.shape[:2]
        m = max(w, h)
        ratio = 112.0 / m
        new_w, new_h = int(ratio * w), int(ratio *h)
        assert new_w > 0 and new_h > 0
        img = cv2.resize(img, (new_w, new_h))
        W, H = 112, 112
        top = (H - new_h) // 2
        bottom = (H - new_h) // 2
        if top + bottom + new_h < H:
            bottom += 1

        left = (W - new_w) // 2
        right = (W - new_w) // 2
        if left + right + new_w < W:
            right += 1

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0,0,0])
        
        #img = cv2.resize(img, (self.width, self.height))
        img = img.transpose(2, 0, 1)
        
        if 'ignore' in label1:
            gt1 = cv2.imread(label1,-1)
            gt1 = cv2.resize(gt1, (112, 112), interpolation = cv2.INTER_NEAREST)
        else:
            gt1 = cv2.imread(label1,-1)/255
            if len(gt1.shape) == 3:
                gt1 = gt1[:,:,0]
            gt1 = cv2.resize(gt1, (new_w, new_h), interpolation = cv2.INTER_NEAREST)
            gt1 = cv2.copyMakeBorder(gt1, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)
            #gt1 = cv2.resize(gt1, (self.label_width, self.label_height), interpolation = cv2.INTER_NEAREST)  

        if 'ignore' in label2:
            gt2 = cv2.imread(label2,-1)
            gt2 = cv2.resize(gt2, (112, 112), interpolation = cv2.INTER_NEAREST)
        else:
            gt2 = cv2.imread(label2,-1)/255
            if len(gt2.shape) == 3:
                gt2 = gt2[:,:,0]
            gt2 = cv2.resize(gt2, (new_w, new_h), interpolation = cv2.INTER_NEAREST)
            gt2 = cv2.copyMakeBorder(gt2, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)
        #gt2 = cv2.resize(gt2, (self.label_width, self.label_height), interpolation = cv2.INTER_NEAREST) 
        cv2.imwrite('gt1.png',gt1)
        cv2.imwrite('gt2.png',gt2)
        imgg = cv2.imread(path).astype(np.float32)
        cv2.imwrite('img.png',imgg)

        return img, gt1, gt2

    def __len__(self):
        return len(self.imgs)
