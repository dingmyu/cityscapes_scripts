import os
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils2 import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, simple_group_split
from torchE.D import dist_init, average_gradients, DistModule
from tensorboardX import SummaryWriter

import segdata as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, poly_learning_rate, intersectionAndUnion
import cv2

# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--train_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/train.txt', help='train list')
    parser.add_argument('--val_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt', help='val list')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--syncbn', type=int, default=1, help='adopt syncbn or not')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--crop_h', type=int, default=473, help='train crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='train crop size w')
    parser.add_argument('--scale_min', type=float, default=0.5, help='minimum random scale')
    parser.add_argument('--scale_max', type=float, default=2.0, help='maximum random scale')
    parser.add_argument('--rotate_min', type=float, default=-10, help='minimum random rotate')
    parser.add_argument('--rotate_max', type=float, default=10, help='maximum random rotate')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='loss weight for aux branch')

    parser.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--bn_group', type=int, default=16, help='group number for sync bn')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size for validation during training, memory and speed tradeoff')
    parser.add_argument('--base_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--power', type=float, default=0.9, help='power in poly learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=10, help='model save step (default: 10)')
    parser.add_argument('--save_path', type=str, default='tmp', help='model and summary save path')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--weight', type=str, default='', help='path to weight (default: none)')
    parser.add_argument('--evaluate', type=int, default=0, help='evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend')
    parser.add_argument('--port', default='23456', type=str)
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


def load_state(path, model, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        logger.info("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(path))


def main():
    global args, logger, writer
    args = get_parser().parse_args()
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    rank, world_size = dist_init(args.port)
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    #if len(args.gpu) == 1:
    #   args.syncbn = False
    if rank == 0:
        logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.crop_h-1) % 8 == 0 and (args.crop_w-1) % 8 == 0
    assert args.net_type in [0, 1, 2, 3]

    if args.bn_group == 1:
        args.bn_group_comm = None
    else:
        assert world_size % args.bn_group == 0
        args.bn_group_comm = simple_group_split(world_size, rank, world_size // args.bn_group)

    if rank == 0:
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

    if args.net_type == 0:
        from pspnet import PSPNet
        model = PSPNet(backbone=args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, syncbn=args.syncbn, group_size=args.bn_group, group=args.bn_group_comm).cuda()
    elif  args.net_type in [1, 2, 3]:
        from pspnet_div4 import PSPNet
        model = PSPNet(backbone=args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, syncbn=args.syncbn, group_size=args.bn_group, group=args.bn_group_comm, net_type=args.net_type).cuda()
    logger.info(model)
    # optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # newly introduced layer with lr x10
    if args.net_type == 0:
        optimizer = torch.optim.SGD(
            [{'params': model.layer0.parameters()},
             {'params': model.layer1.parameters()},
             {'params': model.layer2.parameters()},
             {'params': model.layer3.parameters()},
             {'params': model.layer4.parameters()},
       #      {'params': model.ppm.parameters(), 'lr': args.base_lr * 10},
			 {'params': model.conv6.parameters(), 'lr': args.base_lr * 10},
             {'params': model.cls.parameters(), 'lr': args.base_lr * 10}],
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #model = torch.nn.DataParallel(model).cuda()
    model = DistModule(model)
    #if args.syncbn:
    #    from lib.syncbn import patch_replication_callback
    #    patch_replication_callback(model)
    cudnn.enabled = True
    cudnn.benchmark = True
    criterion = nn.NLLLoss(ignore_index=args.ignore_label).cuda()

    if args.weight:
        def map_func(storage, location):
            return storage.cuda()
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=map_func)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        load_state(args.resume, model, optimizer)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transforms.Compose([
        #transforms.RandScale([args.scale_min, args.scale_max]),
        transforms.Resize((args.crop_h, args.crop_w)),
        transforms.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transforms.RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        #transforms.Crop([args.crop_h, args.crop_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    train_data = datasets.SegData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    if args.evaluate:
        val_transform = transforms.Compose([
            transforms.Crop([args.crop_h, args.crop_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        val_data = datasets.SegData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, args.zoom_factor, args.batch_size, args.aux_weight)
        if rank == 0:
            writer.add_scalar('loss_train', loss_train, epoch)
            writer.add_scalar('mIoU_train', mIoU_train, epoch)
            writer.add_scalar('mAcc_train', mAcc_train, epoch)
            writer.add_scalar('allAcc_train', allAcc_train, epoch)
        # write parameters histogram costs lots of time
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        if epoch % args.save_step == 0 and rank == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            #if epoch / args.save_step > 2:
            #    deletename = args.save_path + '/train_epoch_' + str(epoch - args.save_step*2) + '.pth'
            #    os.remove(deletename)
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, args.classes, args.zoom_factor)
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('mIoU_val', mIoU_val, epoch)
            writer.add_scalar('mAcc_val', mAcc_val, epoch)
            writer.add_scalar('allAcc_val', allAcc_val, epoch)


def train(train_loader, model, criterion, optimizer, epoch, zoom_factor, batch_size, aux_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    end = time.time()
    for i, (input, target, target1) in enumerate(train_loader):
        # to avoid bn problem in ppm module with bin size 1x1, sometimes n may get 1 on one gpu during the last batch, so just discard
        # if input.shape[0] < batch_size:
        #     continue
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = args.epochs * len(train_loader)
        if args.net_type == 0:
            index_split = 4
        elif args.net_type in [1, 2, 3]:
            index_split = 5
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        output, output1 = model(input_var)

        if zoom_factor != 8:
            h = int((target.size()[1]-1) / 8 * zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * zoom_factor + 1)
            # 'nearest' mode doesn't support downsampling operation and while 'bilinear' mode is also fine
            target = F.upsample(target.unsqueeze(1).float(), size=(h, w), mode='bilinear').squeeze(1).long()
            target1 = F.upsample(target1.unsqueeze(1).float(), size=(h, w), mode='bilinear').squeeze(1).long()
        target = target.data.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        target1 = target1.data.cuda(async=True)
        target1_var = torch.autograd.Variable(target1)
        main_loss = criterion(output, target_var) / world_size + criterion(output1, target1_var) / world_size
        loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        output = output.data.max(1)[1].cpu().numpy()
        cv2.imwrite('out.png', output[0]*255)
        target = target.cpu().numpy()
        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        reduced_loss = loss.data.clone()
        reduced_main_loss = main_loss.data.clone()
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_main_loss)

        main_loss_meter.update(reduced_main_loss[0], input.size(0))
        loss_meter.update(reduced_loss[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if rank == 0:
            if (i + 1) % args.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs, i + 1, len(train_loader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              main_loss_meter=main_loss_meter,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if rank == 0:
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, classes, zoom_factor):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        target = target.cuda()#non_blocking=True)
        output = model(input)
        if zoom_factor != 8:
            output = F.upsample(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        output = output.data.max(1)[1].cpu().numpy()
        target = target.cpu().numpy()
        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()

