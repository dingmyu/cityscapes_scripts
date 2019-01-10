import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lanenet
import os
import torchvision as tv
from torch.autograd import Variable
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/share/dingmingyu/cityscapes/instance_list.txt',                         help='the test image list', type=str)
parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/dingmingyu/workspace/instance_seg_pytorch/pic/',
                        help='the test image dir', type=str)
parser.add_argument('--model_path', dest='model_path', default='checkpoints/020_checkpoint.pth.tar',
                        help='the test model', type=str)

def main():
    global args
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(args.model_path)['state_dict']
    model.load_state_dict(state)
    model.eval()    

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    f = open(args.img_list)
    ni = 0
    for line in f:
        if ni > 100:
            break
        line = line.strip()
        arrl = line.split(" ") 
        image = cv2.imread(arrl[0]).astype(np.float32)
        
        img = image[:,:,:3]
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

        image = image.transpose(2, 0, 1)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        print image.shape
        image = torch.from_numpy(image).unsqueeze(0)
        image = Variable(image.float().cuda(0), volatile=True)
        if 'car' in arrl[0]:
            output = model(image)[1]
            print('car', str(ni))
        else:
            output = model(image)[0]
            print('person', str(ni))
        #print output.size()
    	#prob = output.data[0].max(0)[1].cpu().numpy()
        #print prob.max(),prob.shape
        output = torch.nn.functional.softmax(output[0],dim=0)
        prob = output.data.cpu().numpy()
#       prob = output.data[0].max(0)[1].cpu().numpy()
#        print output.size()
        #print output.max(),type(output)
    
        #print prob,prob.shape
        prob = (prob[1] >= 0.6)
        

#        output = torch.nn.functional.softmax(output[0],dim=0)
#        prob = output.data.cpu().numpy() 
#        print prob[1].max(),prob[2].max(),prob[3].max(),prob[4].max(),prob.shape 

        probAll = np.zeros((prob.shape[0], prob.shape[1], 3), dtype=np.float)
        probAll[:,:,0] += prob # line 1
        probAll[:,:,1] += prob # line 2
        probAll[:,:,2] += prob # line 3


        probAll = np.clip(probAll * 255, 0, 255)

        probAll = cv2.resize(probAll, (112,112), interpolation = cv2.INTER_NEAREST)

        ni = ni + 1
        test_img = np.clip(probAll, 0, 255).astype('uint8')
        image = cv2.imread(arrl[0])
        cv2.imwrite(args.img_dir + 'test_' + str(ni) + '_ori.png', image)
        cv2.imwrite(args.img_dir + 'test_' + str(ni) + '_lane.png', test_img)
        print('write img: ' + str(ni+1))
    f.close()

if __name__ == '__main__':
    main()

