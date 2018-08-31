import random
import cv2
import os
import unittest
from data import *
from utils.augmentations import SSDAugmentation, SSDWiderAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from easydict import EasyDict as edict

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

args = edict()
args.dataset = 'VOC'
args.dataset_root = VOC_ROOT
args.basenet = "vgg16_reducedfc.pth"
args.batch_size = 16
args.resume = None
args.start_iter = 0
args.num_workers = 4
args.cuda = True
args.lr = 1e-3
args.momentum=0.9
args.weight_decay=5e-4
args.gamma=0.1
args.visdom=False
args.save_folder="weights/"


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = voc
dataset = VOCDetection(root=args.dataset_root,
    image_sets=[('2007', 'trainval')],
    transform=SSDAugmentation(cfg['min_dim'],
    MEANS))

def get_net():
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    return ssd_net

def show_img(data, gt_boxes):
    img = data
    # print(gt_boxes)
    # img = (img + 122 / 255)
    img_ = (img/122 + 1.0) / 2
    img_ = np.ascontiguousarray(img_)
    # print(img_)
    # print(img_.dtype)
    for box in gt_boxes:
        box = box * 300
        classId = int(box[-1])
        xlb, ylb = box[0], box[1]
        xrt, yrt = box[2], box[3]
        cv2.rectangle(img_, (int(xlb), int(ylb)),
            (int(xrt), int(yrt)), (255, 0, 0), thickness=1)
    return img_
        # bottomLeftCornerOfText = (int(xlb), int(ylb))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, "cat id {}".format(classId), 
        #     bottomLeftCornerOfText, font, 1, 
        #     (200, 200, 200), 2, cv2.LINE_AA)
    # cv2.imshow("data", img_)
    # cv2.waitKey(0)

def gbr2rgb(img_np):
    pass

from data import wider_face as wf
# only do voc dataset
class test(unittest.TestCase):
    def test4(self):
        cfg = wider
        dataset = wf.WiderDetection(
            root="/media/bigtree/DATA/data_ubuntu/wider_face",
            transform=SSDWiderAugmentation(cfg['min_dim'], MEANS)
            )
        epoch_size = len(dataset)
        data_loader = data.DataLoader(dataset, args.batch_size,
            num_workers=1, shuffle=True, 
            collate_fn=detection_collate, 
            pin_memory=True)
        # print(list(map(lambda x: len(x), dataset._bboxes)))
        batch_iterator = iter(data_loader)
        
        ssd_net = build_ssd('train', cfg['min_dim'], 
            cfg['num_classes'])
        net = ssd_net
        if args.cuda:
            # net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True
        net = net.cuda()
        if not args.resume:
            print('Initializing weights...')
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 
            True, 0, True, 3, 0.5, False, args.cuda)
        '''
            run one batch of batch to see if things work
        '''
        images, targets = next(batch_iterator)
        # targets.data[0] = torch.tensor()
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)\
                for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) \
                for ann in targets]
        t0 = time.time()
        out = net(images)
        optimizer.zero_grad()
        # print(out[0].shape, out[1].shape, out[2].shape)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        print("timer: {} seconds loss {} loc loss {}"
            " conf loss {}".format(t1-t0, loss.data[0], 
            loss_l.data[0], loss_c.data[0]))


    def test3(self):
        np.set_printoptions(suppress=True)
        cfg = wider
        dataset = wf.WiderDetection(
            root="/media/bigtree/DATA/data_ubuntu/wider_face",
            # transform=SSDWiderAugmentation(cfg['min_dim'], MEANS)
            transform=None
        )
        len_ds = len(dataset)
        rand_ind = random.choice(list(range(len_ds)))
        im, gt = dataset[rand_ind]
        img_np = np.transpose(im.numpy(), (1, 2, 0)) 
        # print(img_np.shape)
        img_np = np.ascontiguousarray(img_np)
        # print(gt)
        for box in gt:
            xlb, ylb = box[0], box[1]
            xrt, yrt = box[2], box[3]
            cv2.rectangle(img_np, (int(xlb), int(ylb)),
                (int(xrt), int(yrt)), (122, 122, 122),
                thickness=1)

        cv2.imshow('img', img_np)
        cv2.waitKey(0)

    def test2(self):
        cfg = wider
        dataset = wf.WiderDetection(
            root="/media/bigtree/DATA/data_ubuntu/wider_face",
            transform=SSDWiderAugmentation(cfg['min_dim'], MEANS)
            )
        epoch_size = len(dataset)
        data_loader = data.DataLoader(dataset, args.batch_size,
            num_workers=1, shuffle=True, 
            collate_fn=detection_collate, 
            pin_memory=True)
        batch_iterator = iter(data_loader)
        img, targets = next(batch_iterator)
        img1 = img[0].numpy()
        # print(targets[0].numpy())
        # print(img1.shape)
        img1 = np.transpose(img1, (1, 2, 0))
        img1 = (img1/122 + 1.0) / 2
        cv2.imshow('img', img1)
        cv2.waitKey(0)
        # print(img1.shape)
        # img1 = (img1/122 + 1.0) / 2
        show_img_ = np.ascontiguousarray(img1)
        print(targets[0].numpy()*300)
        for box in targets[0].numpy():
            box = box * 300
            xlb, ylb = box[0], box[1]
            xrt, yrt = box[2], box[3]
            cv2.rectangle(show_img_, (int(xlb), int(ylb)),
                (int(xrt), int(yrt)), (0, 0, 0),
                thickness=1)
        cv2.imshow('show_img', show_img_)
        cv2.waitKey(0)

        print(img.shape)


    def test1(self):
        ssd_net = get_net()
        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True
        basenet_name = os.path.join(args.save_folder, args.basenet)
        vgg_weights = torch.load(basenet_name)
        print("loading base net from {}".format(basenet_name))
        ssd_net.vgg.load_state_dict(vgg_weights)
        net = net.cuda()
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 
            True, 0, True, 3, 0.5, False, args.cuda)

        loc_loss = 0
        conf_loss = 0
        epoch = 0
        
        print('Loading the dataset...')
    
        epoch_size = len(dataset) // args.batch_size
        print("len dataset {} batchsize {}".format(
            epoch_size, args.batch_size))
        print('Training SSD on: {} epoch size {}'.format(
            dataset.name, epoch_size))
        print('Using the specified args:')
        print(args)

        step_index = 0

        data_loader = data.DataLoader(dataset, args.batch_size,
            num_workers=args.num_workers,
            shuffle=True, collate_fn=detection_collate,
            pin_memory=True)

        batch_iterator = iter(data_loader)
        print("max iter {}".format(cfg['max_iter']))
        n_iter = 1# cfg['max_iter']

        for iteration in range(0, n_iter):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma,
                    step_index)
            images, targets = next(batch_iterator)
            t0 = time.time()
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) \
                for ann in targets]
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            # print("loss l {} loss c {} loss {}".format(
            #     loss_l, loss_c, loss))

            # # print(images.shape, len(targets))
            # img1 = images[0].numpy()
            # # print(targets[0].numpy())
            # # print(img1.shape)
            # img1 = np.transpose(img1, (1, 2, 0))
            # # img1 = (img1/122 + 1.0) / 2
            # # print(img1)
            # img1 = show_img(img1, targets[0].numpy())
            # cv2.imshow('img', img1)
            # cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()
