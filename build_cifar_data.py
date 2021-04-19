import torchvision
import PIL.Image as Image
import numpy as np
import tqdm
import os

def save_image(np_array, index_list, save_folder, label_name,ext='bmp'):
    for one_img, index in tqdm.tqdm(zip(np_array, index_list)):
        Image.fromarray(one_img).save('%s/%s_%04d.%s'%(save_folder, label_name, index, ext))

save_folder = 'data_folder'
sub_folder_1 = ['sl_data', 'ssl_data']
sub_folder_2 = ['train', 'val']
label_num = 100

sl_folder = '%s/%s'%(save_folder, sub_folder_1[0])
ssl_folder = '%s/%s'%(save_folder, sub_folder_1[1])

dataset = torchvision.datasets.CIFAR10(save_folder, train=True, download=True)
dataset_val = torchvision.datasets.CIFAR10(save_folder, train=False, download=True)

label_name_dist = {j:i for i,j in dataset.class_to_idx.items()}

#load validation dataset
for label_index, label_name in label_name_dist.items():
    save_folder_one = '%s/%s/%s'%(sl_folder, sub_folder_2[1], label_name)
    os.makedirs(save_folder_one)
    one_type_data = dataset_val.data[np.where(np.array(dataset_val.targets) == label_index)]
    save_image(one_type_data, [i for i in range(len(one_type_data))], save_folder_one, label_name)

#load train dataset 
for label_index, label_name in label_name_dist.items():
    save_train = '%s/%s/%s'%(sl_folder, sub_folder_2[0],label_name)
    save_ssl = '%s/%s'%(ssl_folder, label_name)
    os.makedirs(save_train)
    os.makedirs(save_ssl)
    one_type_data = dataset.data[np.where(np.array(dataset.targets) == label_index)]
    np.random.shuffle(one_type_data)
    save_image(one_type_data[0:label_num], [i for i in range(label_num)], save_train, label_name)
    save_image(one_type_data[label_num:],[i for i in range(label_num, len(one_type_data))], save_ssl, label_name)