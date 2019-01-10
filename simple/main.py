import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import *
import lanenet
import os
import torch.nn.functional as F
from loss import *
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--new_length', default=112, type=int)
parser.add_argument('--new_width', default=112, type=int)
parser.add_argument('--label_length', default=112, type=int)
parser.add_argument('--label_width', default=112, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=5, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0




def main():
    global args, best_prec
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    #model.apply(weights_init)
    #params = torch.load('checkpoints/old.pth.tar')
    #model.load_state_dict(params['state_dict'])
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = cross_entropy2d
    #criterion = torch.nn.DataParallel(criterion).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # data transform
    
    train_data = MyDataset('/mnt/lustre/share/dingmingyu/cityscapes/instance_list.txt', args.dir_path, args.new_width, args.new_length,args.label_width,args.label_length)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        print 'epoch: ' + str(epoch + 1)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set

        # remember best prec and save checkpoint

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_name, args.resume)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_seg = AverageMeter()
    lrs = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target1, target2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr = adjust_learning_rate(optimizer, epoch*len(train_loader)+i, args.epochs*len(train_loader))
        lrs.update(lr)
        input = input.float().cuda()
        target1 = target1.long().cuda()
        target2 = target2.long().cuda()



        input_var = torch.autograd.Variable(input)
        target1_var = torch.autograd.Variable(target1)
        target2_var = torch.autograd.Variable(target1)


        x_sem1, x_sem2 = model(input_var)

        loss_seg = criterion(x_sem1, target1_var) + criterion(x_sem2, target2_var)
        #print loss_seg
        loss = loss_seg

        losses_seg.update(loss.data[0], input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_seg {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
                  'Lr {lr.val:.5f} ({lr.avg:.5f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_seg=losses_seg, lr=lrs))



def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, curr_iter, max_iter, power=0.9):
    lr = args.lr * (1 - float(curr_iter)/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
