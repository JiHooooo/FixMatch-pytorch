import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict

import glob
import statistics

from PIL import Image, ImageDraw
import pandas as pd
import tqdm

from utils import net_builder
from datasets.custom_dataset import _label_name, get_transform

parser = argparse.ArgumentParser(description='pytorch classification implement')
parser.add_argument('--pretrained-model',type=str, default='result/final_model.pth')
parser.add_argument('--gpu',type=int, default=0)
#Dataset Configurations
parser.add_argument('--image_size', type=int, default=224,
                    help='image size for model calculation')
parser.add_argument('--normlization', type=bool, default=True,
                    help='whether to use normlization')
parser.add_argument('--img_extension', type=str, default='bmp')
#Backbone Net Configurations
parser.add_argument('--net', type=str, default='SelfModel',
                    help='model name used for training model')
parser.add_argument('--net_from_name', type=bool, default=False)
#The relevant parameter for initializing wideResnet
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)


parser.add_argument('--demo-dir', type=str, default='DemoImage')
parser.add_argument('--demo-save', type=str, default='DemoResult')
parser.add_argument('--evaluation-mode',action='store_true')
parser.add_argument('--amp',action='store_true', help='whether to use AMP')

args = parser.parse_args()
#print(torch.load(args.pretrained_model))

def write_txt(txt_path, txt_info):
    #txt_path: the path where the txt is saved
    #txt_info: list of info [image_name, label, score]
    #image_name:str, label:str, score:float
    with open(txt_path, 'a') as f:
        f.write('name:%s, label:%s, score:%.3f\n'%(txt_info[0], txt_info[1], txt_info[2]))

#gpuを使うがどうか
if args.gpu >= 0:
    device = torch.device('cuda:%d'%(args.gpu))
else:
    device = torch.device('cpu')

#モデル導入
model_builder = net_builder(args.net, 
                            args.net_from_name,
                            None,
                            {'depth': args.depth, 
                            'widen_factor': args.widen_factor,
                            'leaky_slope': args.leaky_slope,
                            'dropRate': args.dropout})
model = model_builder(len(_label_name))
model.eval()
model.to(device)

#学習済みモデルのパラメータを導入する
pretrained_model_file = torch.load(args.pretrained_model, map_location=device)
new_state_dict = OrderedDict()
for k,  v in pretrained_model_file['eval_model'].items():
    #複数GPU学習でモデル名が違うので、調整する必要がある
    if k[:7] == 'module.':
        name = k[7:]
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

time_record = []
#推論画像を導入する
#
if args.evaluation_mode:
    image_paths = []
    image_labels = []
    for single_folder in os.listdir(args.demo_dir):
        if single_folder in _label_name:
            one_folder_path = glob.glob('%s/%s/*.%s'%(args.demo_dir, 
                                                        single_folder, 
                                                        args.img_extension))
            image_paths.extend(one_folder_path)
            image_labels.extend([_label_name.index(single_folder) for 
                                index in range(len(one_folder_path))]
                                )
    confusion_matrix = np.zeros((len(_label_name), len(_label_name)), dtype = np.uint32)
    index_matrix = [name+'_pred' for name in _label_name]
    columns_matrix = [name+'_ground' for name in _label_name]
else:
    image_paths = glob.glob('%s/*.%s'%(args.demo_dir, args.img_extension))

os.makedirs(args.demo_save, exist_ok=True)

#load image augmentation method
val_transforms = get_transform(False, image_size=args.image_size, normalize_flag=args.normlization)

#create the txt file which is used for saving results
txt_result_path = os.path.join(args.demo_save, 'demo_results.txt')
with open(txt_result_path,'w') as f:
    pass

time_record = [[] for _ in range(3)]
#create a folder to save the image results
for label_name_one in _label_name:
    os.makedirs('%s/%s'%(args.demo_save, label_name_one), exist_ok=True)
#main process
for image_index, image_path in tqdm.tqdm(enumerate(image_paths),total=len(image_paths)):
    _, image_name = os.path.split(image_path)
    #画像入力
    img_PIL_ori = Image.open(image_path)
    draw = ImageDraw.Draw(img_PIL_ori)
    time_1 = time.time()
    #モデルに結果を計算する
    input_tensor = val_transforms(img_PIL_ori)
    time_2 = time.time()
    with torch.no_grad():
        if args.amp:
            with torch.cuda.amp.autocast():
                results = nn.Softmax(dim=1)(model(input_tensor.to(device)[None]))
        else:
            results = nn.Softmax(dim=1)(model(input_tensor.to(device)[None]))
    time_3 = time.time()
    #判断種類とスコアを計算する
    predict_y = int(results.data.argmax(axis=1)[0])
    predict_label_name = _label_name[predict_y]
    #predict_label_name = str(predict_y)
    predict_score = results.data[0][predict_y]
    time_4 = time.time()
    time_record[0].append(time_2 - time_1)
    time_record[1].append(time_3 - time_2)
    time_record[2].append(time_4 - time_3)
    #record the results
    
    write_txt(txt_result_path, [image_name, predict_label_name, predict_score])
    draw.text((10,10), '%s: %.4f'%(predict_label_name, predict_score),fill=(255,0,0))
    #print('%s is %s with score of %.4f'%(image_name, predict_label_name, predict_score))
    #save the image results
    if (predict_y != 0) or (predict_y == 0  and predict_score <0.8):
        img_PIL_ori.save('%s/%s/%s'%(args.demo_save, _label_name[predict_y], image_name))
    
    if args.evaluation_mode:
        confusion_matrix[predict_y, image_labels[image_index]] += 1
if args.evaluation_mode:
    pd_matrix = pd.DataFrame(data = confusion_matrix, index = index_matrix,
                            columns = columns_matrix, dtype='int')
    print('The confusion matrix:\n'+str(pd_matrix) +'\n')
    pd_matrix.to_csv(args.demo_save + '/confusion_matrix.csv')

#平均速度を計算する時に、最初の結果を入れない
avergae_time = [statistics.mean(i[1:])*1000 for i in time_record]
print('%d images have been calculated\n \
    the average preprocessing time is %.3f ms\n \
    the average model calculation time is %.3f ms\n \
    the average postrocessing time is %.3f ms\n'%(len(image_paths), avergae_time[0], avergae_time[1], avergae_time[2]))
print('ok')