import os
import shutil
import time
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.mytransforms as mytransforms
import numpy as np
from utils.config import FLAGS
from utils.setlogger import get_logger

saved_path = FLAGS.log_dir
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'train.log'))
best_acc1 = 0
best_acc5 = 0


def main():
    global best_acc1, best_acc5

    traindir = os.path.join(FLAGS.dataset_dir, 'train')
    valdir = os.path.join(FLAGS.dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = mytransforms.Lighting(alphastd=0.1)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            jittering,
            lighting,
            # transforms.ToTensor(),
            # normalize,
    ])
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=mytransforms.MultiCropsTransform(train_transform)
        )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
        num_workers=FLAGS.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=FLAGS.batch_size//2, shuffle=False,
        num_workers=FLAGS.workers, pin_memory=True)
    numberofclass = 1000

    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(depth=FLAGS.depth, num_classes=numberofclass)
    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                momentum=FLAGS.momentum,
                                weight_decay=FLAGS.weight_decay, nesterov=FLAGS.nesterov)
    lr_scheduler = get_lr_scheduler(optimizer, train_loader)



    for epoch in range(0, FLAGS.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler)

        # evaluate on validation set
        acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc5 = acc5

        print('Current best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)
        save_checkpoint({
            'epoch': epoch,
            # 'arch': FLAGS.net_type,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    reso_idx = list(np.random.randint(0, len(FLAGS.resos), FLAGS.num_subnet))
    train_loader.dataset.transform.set_resoidx(reso_idx)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # first do max_width and max_resolution
        max_width = FLAGS.max_width
        model.apply(lambda m: setattr(m, 'width_mult', max_width))
        max_output = model(input[0])
        loss = torch.mean(criterion(max_output, target))
        loss.backward()
        max_output_detach = max_output.detach()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(max_output.data, target, topk=(1, 5))
        losses.update(loss.item(), input[0].size(0))
        top1.update(acc1.item(), input[0].size(0))
        top5.update(acc5.item(), input[0].size(0))

        # do other widths and resolution
        min_width = FLAGS.min_width
        width_mult_list = [min_width]
        sampled_width = list(np.random.uniform(min_width, max_width, FLAGS.num_subnet-1))
        width_mult_list.extend(sampled_width)
        sub_idx = 1
        for width_mult in sorted(width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            output = model(input[sub_idx])
            sub_idx += 1
            loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1),
                                                             F.softmax(max_output_detach, dim=1))
            loss.backward()

        optimizer.step()
        lr_scheduler.step()
        reso_idx = list(np.random.randint(0, len(FLAGS.resos), FLAGS.num_subnet))
        train_loader.dataset.transform.set_resoidx(reso_idx)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            logger.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, FLAGS.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    logger.info('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, FLAGS.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, FLAGS.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, FLAGS.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = FLAGS.log_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


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


def get_lr_scheduler(optimizer, trainloader):
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.epochs*len(trainloader))
    else:
        raise NotImplementedError('LR scheduler not implemented.')
    return lr_scheduler


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if args.dataset.startswith('cifar'):
#         lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
#     elif args.dataset == ('imagenet'):
#         if args.epochs == 300:
#             lr = args.lr * (0.1 ** (epoch // 75))
#         else:
#             lr = args.lr * (0.1 ** (epoch // 30))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


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


if __name__ == '__main__':
    main()