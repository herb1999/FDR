import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNet(Dataset):

    def __init__(self, root='', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, autoaug=True):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.normpath(os.path.join(root, 'miniimagenet/images'))
        self.SPLIT_PATH = os.path.normpath(os.path.join(root, 'miniimagenet/split'))

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        image_size = 224
        if autoaug is False:
            #do not use autoaug.
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711]) ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize([image_size, image_size]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711]) ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            #use autoaug.
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711]) ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize([image_size, image_size]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711]) ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, data2label, index_path):
        #select from txt file, and make cooresponding mampping.
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.normpath(os.path.join(self.IMAGE_PATH, i))
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        #select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


class MiniImageNet_concate(Dataset):
    def __init__(self, train,x1,y1,x2,y2):
        
        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            
        
        self.data=x1+x2
        self.targets=y1+y2
        print(len(self.data),len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

if __name__ == '__main__':
    trainset = MiniImageNet(root='D:\code\TEEN-main\data', train=True,
                                         index_path='D:\code\TEEN-main\data\index_list\mini_imagenet\session_2_shot_5.txt')