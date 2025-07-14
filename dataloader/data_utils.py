import os.path

import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from torch.utils.data import Dataset
import csv


def get_id2label(path):
    """
    从文件读取数据，生成 id2label 字典。
    """
    id2label = {}

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                key, value = parts
                id2label[int(key)] = value  # 将 key 转换为整数

    return id2label


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        if args.project == 'mine':
            import dataloader.cifar100.cifar_clip as Dataset
        else:
            import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        # args.shot = 5
        args.sessions = 9

    if args.dataset == 'cub200':
        if args.project == 'mine':
            import dataloader.cub200.cub200_clip as Dataset
        else:
            import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        # args.shot = 5
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        if args.project == 'mine':
            import dataloader.miniimagenet.miniimagenet_clip as Dataset
        else:
            import dataloader.miniimagenet.miniimagenet as Dataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        # args.shot = 5
        args.sessions = 9

    if args.dataset == 'in1k_fscil':
        import dataloader.in1k_fscil.in1k_fscil as Dataset
        args.base_class = 800
        args.num_classes = 1000
        args.way = 20
        # args.shot = 1  # 默认 1-shot，可选3、5 shot
        args.sessions = 11  # 10 个增量 session + session 0

    args.Dataset = Dataset
    return args


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader


def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'in1k_fscil':
        train_csv = os.path.join(args.dataroot, "session_0.csv")
        test_csv = os.path.join(args.dataroot, "test_session_0.csv")
        wnid_map_file = os.path.join(args.dataroot, "imagenet_wnid_map.txt")
        trainset = args.Dataset.ImageNetFSCIL(train=True, index_path=train_csv,
                                              wnid_map_file=wnid_map_file)
        testset = args.Dataset.ImageNetFSCIL(train=False, index_path=test_csv,
                                             wnid_map_file=wnid_map_file)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args, session):
    txt_path = os.path.join(args.dataroot, 'index_list', args.dataset,
                            "session_" + str(session + 1) + '_shot_' + str(args.shot) + '.txt')

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        print('loading index from ' + txt_path)
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_new)
    if args.dataset == 'in1k_fscil':
        train_csv = os.path.join(args.dataroot, f"session_{session}_{args.shot}shot.csv")
        test_csv = os.path.join(args.dataroot, f"test_session_{session}.csv")
        wnid_map_file = os.path.join(args.dataroot, "imagenet_wnid_map.txt")
        trainset = args.Dataset.ImageNetFSCIL(train=True, index_path=train_csv,
                                              wnid_map_file=wnid_map_file)
        testset = args.Dataset.ImageNetFSCIL(train=False, index_path=test_csv,
                                             wnid_map_file=wnid_map_file)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list
