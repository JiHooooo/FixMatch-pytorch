import os
import glob
import random
import logging
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import copy
from .ssl_dataset import SSL_Dataset

_extension = ['jpg','png','bmp']
_label_name = ['airplane','automobile','ship', 'truck']
_model_mean = [0.485,0.456,0.406]
_model_std = [0.229,0.224,0.225]

def get_transform(train=True, image_size=224, crop_ratio=0.1, normalize_flag=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((image_size,image_size)))
    if train:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size*crop_ratio))
        ])
    if normalize_flag:
        transforms_list.extend([
            transforms.ToTensor(), 
            transforms.Normalize(_model_mean, _model_std),
        ])
    else:
        transforms_list.extend([
            transforms.ToTensor(),
        ])
    return transforms.Compose(transforms_list)

class SelfDataset(Dataset):
    """
    SelfDataset returns a pair of image and labels (targets).
    This class supports strong augmentation for Fixmatch,5
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 folder_path,
                 ssl_dataset_flag = False,
                 transforms=None,
                 use_strong_transform=False,
                 strong_transforms=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            folder_path: the folder where the training images are saved

            ##### folder structure
                folder -
                    label1:
                        image1
                        image2
                        ...
                    label2:
                    ...
            ##### 

            ssl_dataset_flag : whether the images which are in the folder have reliable label
            transforms: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(SelfDataset, self).__init__()
        self.transforms = transforms
        self.ssl_dataset_flag = ssl_dataset_flag
        self.num_classes = len(_label_name)
        self.label_names = _label_name
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot

        #read all image path
        if isinstance(folder_path, str):
            image_path_list, image_label_list = \
                self.load_image_label_from_folder(folder_path)
        elif isinstance(folder_path, list):
            image_path_list = []
            image_label_list = []
            for folder_path_one in folder_path:
                one_folder_image_path_list, one_folder_image_label_list = self.load_image_label_from_folder(folder_path_one)
                image_path_list.extend(one_folder_image_path_list)
                image_path_list.extend(one_folder_image_label_list)
        else:
            raise TypeError('The type of folder path should be str or list')

        if len(image_path_list) == 0:
            raise ValueError("Don't find suitable image file")

        if 'shuffle_seed' in kwargs.keys() and 'lb_image_num' in kwargs.keys() and not self.ssl_dataset_flag:
            
            self.image_path_list, self.image_label_list = \
                self.split_lb_ulb(image_path_list, image_label_list, 
                    seed=kwargs['shuffle_seed'], lb_num=kwargs['lb_image_num'],
                    lb_flag=not use_strong_transform)

        elif self.ssl_dataset_flag:
            self.image_label_list = image_label_list
            self.image_path_list = image_path_list
        else:
            self.image_path_list, self.image_label_list = \
                self.split_lb_ulb(image_path_list, image_label_list)
        
        if use_strong_transform:
            if strong_transforms is None:
                self.strong_transforms = copy.deepcopy(transforms)
                self.strong_transforms.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transforms = strong_transforms

    @staticmethod
    def split_lb_ulb(image_path_list, image_label_list, seed=None, lb_num=None, lb_flag=True):
        # image_path_list [[image_path_of_typeA],[...], ...]
        # image_label_list [[image_label_of_typeA],[...], ...]
        total_image_pathes = []
        total_image_labels = []
        for image_pathes, image_labels in zip(image_path_list, image_label_list):
            if lb_num is None or seed is None or lb_num <0 or lb_num > len(image_pathes):
                total_image_pathes.extend(image_pathes)
                total_image_labels.extend(image_labels)
            else:
                random.seed(seed)
                random.shuffle(image_pathes)
                random.seed(seed)
                random.shuffle(image_labels)
                if lb_flag:
                    total_image_pathes.extend(image_pathes[:lb_num])
                    total_image_labels.extend(image_labels[:lb_num])
                else:
                    total_image_pathes.extend(image_pathes[lb_num:])
                    total_image_labels.extend(image_labels[lb_num:])
        return total_image_pathes, total_image_labels
    
    def load_image_label_from_folder(self, folder_path):

        if not self.ssl_dataset_flag:
            sub_folder_list = os.listdir(folder_path)
            image_path_list = []
            image_label_list = []
            
            for label_folder in sub_folder_list:
                if label_folder in self.label_names:
                    image_pathes_one_folder = self.load_image_of_one_folder('%s/%s'%(folder_path, label_folder))
                    image_path_list.append(image_pathes_one_folder)
                    image_label_list.append([self.label_names.index(label_folder) for _ in image_pathes_one_folder])
        else:
            image_path_list = self.load_image_of_one_folder(folder_path)
            image_label_list = [-1 for _ in image_path_list]

        return image_path_list, image_label_list

    @staticmethod
    def load_image_of_one_folder(folder_path):
        image_pathes = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if os.path.splitext(file_name)[1][1:] in _extension:
                    image_pathes.append('%s/%s'%(root, file_name))
        return image_pathes


    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.image_label_list is None:
            target = None
        else:
            target_ = self.image_label_list[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
        
        #set augmented images
        #load image
        image_path = self.image_path_list[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transforms is None:
            return transforms.ToTensor()(image), target
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            img_w = self.transforms(image)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return img_w, self.strong_transforms(image), target

    
    def __len__(self):
        return len(self.image_path_list)

class SelfDataset_fold(SelfDataset):

    def __init__(self,
                 csv_path,
                 ssl_dataset_flag = False,
                 transforms=None,
                 use_strong_transform=False,
                 strong_transforms=None,
                 onehot=False,
                 train_flag=True,
                 fold_num = 0,
                 *args, **kwargs):
        """
        Args
            csv_path: the csv file where the training images are saved

            ##### folder structure
                folder -
                    label1:
                        image1
                        image2
                        ...
                    label2:
                    ...
            ##### 

            ssl_dataset_flag : whether the images which are in the folder have reliable label
            transforms: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        self.transforms = transforms
        self.ssl_dataset_flag = ssl_dataset_flag
        self.num_classes = len(_label_name)
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot

        #read all image path
        df_info = pd.read_csv(csv_path)
        print('the label name : ' + str(_label_name))
        
        self.label_names = [str(i) for i in _label_name]
        if train_flag:
            selected_df_info = df_info[df_info['fold'] != fold_num]
        else:
            selected_df_info = df_info[df_info['fold'] == fold_num]

        #delete the image whose label is not included in label name
        selected_df_info = selected_df_info[selected_df_info['label'].isin(self.label_names)]
        image_path_list = []
        image_label_list = []
        for label_one in self.label_names:
            selected_df_info_one_label = selected_df_info[selected_df_info['label'] == label_one]
            image_path_list.append(list(selected_df_info_one_label['image_path']))
            image_label_list_ori = list(selected_df_info_one_label['label'])
            image_label_list.append([self.label_names.index(i) for i in image_label_list_ori])

        if len(image_path_list) == 0:
            raise ValueError("Don't find suitable image file")

        if 'shuffle_seed' in kwargs.keys() and 'lb_image_num' in kwargs.keys() and not self.ssl_dataset_flag:
            
            self.image_path_list, self.image_label_list = \
                self.split_lb_ulb(image_path_list, image_label_list, 
                    seed=kwargs['shuffle_seed'], lb_num=kwargs['lb_image_num'],
                    lb_flag=not use_strong_transform)

        elif self.ssl_dataset_flag:
            self.image_label_list = image_label_list
            self.image_path_list = image_path_list
        else:
            
            self.image_path_list, self.image_label_list = \
                self.split_lb_ulb(image_path_list, image_label_list)
        
        if use_strong_transform:
            if strong_transforms is None:
                self.strong_transforms = copy.deepcopy(transforms)
                self.strong_transforms.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transforms = strong_transforms

class SelfDataset_multi(Dataset):
    def __init__(self, csv_path,transforms = None, 
                seed=0, lb_num=0, lb_flag=True):
        image_path_list, image_label_list = self.Image_Info_from_df(csv_path)
        self.image_path_list, self.image_label_list = self.split_lb_ulb(
            image_path_list, image_label_list, 
            seed = seed, lb_num=lb_num, lb_flag=lb_flag
        )
        self.transforms = transforms
        self.label_num = len(_label_name)
        self.label_names = _label_name
        if not lb_flag:
            self.strong_transforms = copy.deepcopy(transforms)
            self.strong_transforms.transforms.insert(0, RandAugment(3,5))
            self.use_strong_transform = True
        else:
            self.use_strong_transform = False
    def __len__(self):
        return len(self.image_path_list)


    @staticmethod
    def Image_Info_from_df(df_path):
        try:
            df = pd.read_csv(df_path,encoding="cp932")
        except:
            df = pd.read_csv(df_path,encoding="utf-8")
            logging.info('load csv with utf-8 encoding method')
        else:
            logging.info('load csv with cp932 encoding method')
        image_path_list = []
        image_label_list = []
        for index in range(len(df)):
            #input image name
            image_info_one = [df.iloc[index]['image_path'],]
            for label in self.label_names:
                image_info_one.append(int(df.iloc[index][label]))                                 
            image_path_list.append(image_info_one[0])
            image_label_list.append(image_info_one[1:])
        return image_path_list, image_label_list
    
    @staticmethod
    def split_lb_ulb(image_path_list, image_label_list, seed=0, lb_num=0, lb_flag=True):
        if lb_num <= 0 or lb_num >= len(image_path_list):
            output_image_path_list = image_path_list
            output_image_label_list = image_label_list
        else:
            random.seed(seed)
            random.shuffle(image_path_list)
            random.seed(seed)
            random.shuffle(image_label_list)
            if lb_flag:
                output_image_path_list = image_path_list[:lb_num]
                output_image_label_list = image_label_list[:lb_num]
            else:
                output_image_path_list = image_path_list[lb_num:]
                output_image_label_list = image_label_list[lb_num:]
        return output_image_path_list, output_image_label_list        

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)

        if self.image_label_list is None:
            target = np.zeros(self.label_num, dtype=np.float32)
        else:
            target = self.image_label_list[idx]
        
        labels = np.zeros(self.label_num, dtype=np.float32)
        for index_label in range(self.label_num):
            if target[index_label] > 0:
                labels[index_label] = 1
        labels = torch.from_numpy(labels)

        if self.transforms is None:
            return transforms.ToTensor()(image), labels
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            img_w = self.transforms(image)
            if not self.use_strong_transform:
                return img_w, labels
            else:
                return img_w, self.strong_transforms(image), labels

