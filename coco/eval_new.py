import os
import cv2
import time
import logging
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import segdata as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, intersectionAndUnion, check_makedirs, colorize
cv2.ocl.setUseOpenCL(False)


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')
    parser.add_argument('--data_root', type=str, default='', help='data root')
    parser.add_argument('--val_list1', type=str, default='/mnt/lustre/dingmingyu/workspace/instance_seg/testlist.txt', help='val list')
    parser.add_argument('--split', type=str, default='val', help='split in [train, val and test]')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=18, help='layers number of based resnet')
    parser.add_argument('--classes', type=int, default=2, help='number of classes')
    parser.add_argument('--base_size1', type=int, default=1, help='based size for scaling')
    parser.add_argument('--crop_h', type=int, default=233, help='validation crop size h')
    parser.add_argument('--crop_w', type=int, default=233, help='validation crop size w')
    parser.add_argument('--zoom_factor', type=int, default=4, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--scales', type=float, default=[1.0], nargs='+', help='evaluation scales')
    parser.add_argument('--has_prediction', type=int, default=0, help='has prediction already or not')

    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=1, help='data loader workers')
    parser.add_argument('--model_path', type=str, default='exp/drivable/res101_psp_coarse_fine/model/train_epoch_120.pth', help='evaluation model path')
    return parser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser().parse_args()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.crop_h - 1) % 8 == 0 and (args.crop_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transforms.Compose([transforms.Resize((args.crop_h, args.crop_w)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    
    val_data1 = datasets.SegData(split=args.split, data_root=args.data_root, data_list=args.val_list1, transform=val_transform)
    val_loader1 = torch.utils.data.DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    



    from pspnet import PSPNet
    model = PSPNet(backbone = args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, use_softmax=False, use_aux=False, pretrained=False, syncbn=False).cuda()

    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    cv2.setNumThreads(0)




    fff = open('../testlist.txt').readlines()
    flag = []
    for i, line in enumerate(fff):
        if i > 100:
            break
        img = line.strip().split()[0]
        img = cv2.imread(img)
        cv2.imwrite('result1/result_%d_ori.png' % i, img)
        
        
    validate(val_loader1, val_data1.data_list, model, args.classes, mean, std, args.base_size1, args.crop_h, args.crop_w, flag)

def validate(val_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, flag):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, label1, label2) in enumerate(val_loader):
        if i > 100:
            break
        data_time.update(time.time() - end)
        input = input.float().cuda(async=True)
        input_var = torch.autograd.Variable(input)
        output, output1 = model(input_var)
        output = output1
        output = output.squeeze(0).data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        print (output.shape)
        cv2.imwrite('result1/result_%d.png' % i, output[:,:,1]*255)

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')




if __name__ == '__main__':
    main()
