import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.wideresnet_randwidth import WideResNet_randwidth
from models.pyramidnet_randwidth import PyramidNet_randwidth
import models.resnet_randdepth as resnet_randdepth
from utils.setlogger import get_logger

from utils.config import FLAGS
import numpy as np
import random

best_prec1 = 0

logpath = FLAGS.log_dir
if not os.path.exists(logpath):
    os.makedirs(logpath)
logger = get_logger(os.path.join(logpath, 'train.log'))

def main():
    global best_prec1

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                            (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': FLAGS.workers, 'pin_memory': True}
    assert(FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)

    # create model
    if FLAGS.model == 'wideresnet':
        model = WideResNet_randwidth(depth=FLAGS.depth, num_classes=FLAGS.dataset == 'cifar10' and 10 or 100,
                                widen_factor=FLAGS.widen_factor, dropRate=0)
    elif FLAGS.model == 'pyramidnet':
        model = PyramidNet_randwidth(dataset=FLAGS.dataset, depth=200, alpha=240, num_classes=100, bottleneck=True)
    elif FLAGS.model == 'resnet_randdepth':
        model = resnet_randdepth.resnet110_cifar(num_classes=FLAGS.dataset == 'cifar10' and 10 or 100)
    else:
        raise NotImplementedError('model type not implemented.')

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            print("=> loading checkpoint '{}'".format(FLAGS.resume))
            checkpoint = torch.load(FLAGS.resume)
            FLAGS.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(FLAGS.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(FLAGS.resume))

    # cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                weight_decay=FLAGS.weight_decay)

    # cosine learning rate
    scheduler = get_lr_scheduler(optimizer, train_loader)

    if FLAGS.test_only:
        ckpt = torch.load(FLAGS.pretrained)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print('Load pretrained weights from ', FLAGS.pretrained)
        acc1, acc5 = validate(val_loader, model, criterion, 0)
        print('Top-1 and 5 accuracy:', acc1, acc5)
        return

    for epoch in range(FLAGS.start_epoch, FLAGS.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            best_prec5 = prec5
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    logger.info('Best accuracy: Top1:{} Top5:{}'.format(best_prec1, best_prec5))

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        # GradAug training
        if FLAGS.min_width > 0:  # randwidth
            max_width = FLAGS.max_width
            min_width = FLAGS.min_width
            width_mult_list = [min_width]
            sampled_width = list(np.random.uniform(min_width, max_width, FLAGS.num_subnet-1))
            width_mult_list.extend(sampled_width)
            model.apply(lambda m: setattr(m, 'width_mult', max_width))
        else:  # randdepth
            model.apply(lambda m: setattr(m, 'fullnet', True))
        max_output = model(input.cuda(non_blocking=True))
        loss = criterion(max_output, target)
        loss.backward()
        prec1, prec5 = accuracy(max_output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        max_output_detach = max_output.detach()
        if FLAGS.min_width > 0:  # randwidth
            for width_mult in sorted(width_mult_list, reverse=True):
                model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))
                resolution = FLAGS.resos[random.randint(0, len(FLAGS.resos)-1)]
                output = model(F.interpolate(input, (resolution, resolution), mode='bilinear', align_corners=True))
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
                loss.backward()
        else:  # randdepth
            model.apply(lambda m: setattr(m, 'fullnet', False))
            for k in range(3):
                resolution = FLAGS.resos[random.randint(0, len(FLAGS.resos) - 1)]
                output = model(F.interpolate(input, (resolution, resolution), mode='bilinear', align_corners=True))
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1),
                                                                 F.softmax(max_output_detach, dim=1))
                loss.backward()



        # compute gradient and do SGD step
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'LR:{3: .4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time, data_time=data_time,
                      loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            logger.info('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, FLAGS.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    logger.info('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, FLAGS.epochs, top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = logpath
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, logpath + 'model_best.pth.tar')

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_lr_scheduler(optimizer, trainloader):
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 225], gamma=0.1)
    elif FLAGS.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.epochs*len(trainloader))
    else:
        raise NotImplemented('LR scheduler not implemented.')
    return lr_scheduler

if __name__ == '__main__':
    main()
