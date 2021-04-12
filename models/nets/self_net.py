import torch
import torch.nn as nn
#import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor

_input_size = (224,224)
_model_mean = [0.485,0.456,0.406]
_model_std = [0.229,0.224,0.225]
_label_name = ['A','B','C','bad']
_label_num = len(_label_name)
#_label_name = ['1_Normal','2_Kamemushi']
_model_name = 'resnet50'
#_label_name = ['normal','okkizu','nama1','nama2','nama3', 'hanazami','hiyake','mizuyake','black_point','hana']
# _model_name : resnet50

_gram_layer_info = {
    'resnet18' : {'Layer_name':'layer3', 'Layer_index':['1',]},
    'resnet34' : {'Layer_name':'layer4', 'Layer_index':['1',]},
    'resnet50' : {'Layer_name':'layer4', 'Layer_index':['2',]},
    'resnet101' : {'Layer_name':'layer4', 'Layer_index':['2',]},
    'resnet152' : {'Layer_name':'layer4', 'Layer_index':['2',]},
    'vgg11': {'Layer_name':'features', 'Layer_index':['20',]},
    'vgg13': {'Layer_name':'features', 'Layer_index':['24',]},
    'vgg16': {'Layer_name':'features', 'Layer_index':['30',]},
    'vgg19': {'Layer_name':'features', 'Layer_index':['36',]},
    'vgg11_bn': {'Layer_name':'features', 'Layer_index':['24',]},
    'vgg13_bn': {'Layer_name':'features', 'Layer_index':['34',]},
    'vgg16_bn': {'Layer_name':'features', 'Layer_index':['43',]},
    'vgg19_bn': {'Layer_name':'features', 'Layer_index':['52',]},
}

_Final_layer_info = {
    'resnet18' : {'Layer_name':'fc', 'Layer_index':-1},
    'resnet34' : {'Layer_name':'fc', 'Layer_index':-1},
    'resnet50' : {'Layer_name':'fc', 'Layer_index':-1},
    'resnet101' : {'Layer_name':'fc', 'Layer_index':-1},
    'resnet152' : {'Layer_name':'fc', 'Layer_index':-1},
    'googlenet':{'Layer_name':'fc', 'Layer_index':-1},
    'vgg11': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg13': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg16': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg19': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg11_bn': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg13_bn': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg16_bn': {'Layer_name':'classifier', 'Layer_index':6},
    'vgg19_bn': {'Layer_name':'classifier', 'Layer_index':6},
    'mobilenet_v2':{'Layer_name':'classifier', 'Layer_index':1},
    'densenet121':{'Layer_name':'classifier', 'Layer_index':-1},
    'densenet161':{'Layer_name':'classifier', 'Layer_index':-1},
    'densenet169':{'Layer_name':'classifier', 'Layer_index':-1},
    'densenet201':{'Layer_name':'classifier', 'Layer_index':-1},
    'alexnet':{'Layer_name':'classifier', 'Layer_index':6},
    'inception_v3':{'Layer_name':'fc', 'Layer_index':-1},
    'resnext50_32x4d':{'Layer_name':'fc', 'Layer_index':-1},
    'resnext101_32x8d':{'Layer_name':'fc', 'Layer_index':-1},
    'shufflenet_v2_x0_5':{'Layer_name':'fc', 'Layer_index':-1},
    'shufflenet_v2_x1_0':{'Layer_name':'fc', 'Layer_index':-1},
    'shufflenet_v2_x1_5':{'Layer_name':'fc', 'Layer_index':-1},
    'shufflenet_v2_x2_0':{'Layer_name':'fc', 'Layer_index':-1},
    'mnasnet0_5':{'Layer_name':'classifier', 'Layer_index':1},
    'mnasnet0_75':{'Layer_name':'classifier', 'Layer_index':1},
    'mnasnet1_0':{'Layer_name':'classifier', 'Layer_index':1},
    'mnasnet1_3':{'Layer_name':'classifier', 'Layer_index':1},
}


class CNN(nn.Module):
    def __init__(self, device=None,train_flag=True):
        self.device = device
        self.input_size = _input_size
        if train_flag:
            pretrained_model_flag = True
        else:
            pretrained_model_flag = False
        extractor = models.__dict__[_model_name](pretrained=pretrained_model_flag)
        #extractor = models.__dict__[_model_name](pretrained=False)
        extractor = Change_the_last_layer(extractor, _model_name, _label_num)
        super().__init__()
        self.extractor = extractor

        self.transform = A.Compose([
            A.Resize(height=_input_size[0], width=_input_size[1], p=1.0),
            A.Normalize(p=1.0),
            ToTensorV2(),
            #ToTensor(),
        ])

    
    def forward(self, x):
        return self.extractor(x)

    def predict(self, x):
        x = self.image_process(x)
        with torch.no_grad():
            x = self.forward(x)
        return x
        
    def image_process(self, x):
        transformed = self.transform(image=x)
        x = transformed['image']
        return x[None].to(self.device)
        


def Change_the_last_layer(model, model_name, layer_num):
    Final_layer_info = _Final_layer_info[model_name]
    if Final_layer_info['Layer_index'] >= 0:
        final_module = getattr(model, Final_layer_info['Layer_name'])
        fc_feature = final_module[Final_layer_info['Layer_index']].in_features
        final_module[Final_layer_info['Layer_index']] = nn.Linear(fc_feature, layer_num) 
        setattr(model, Final_layer_info['Layer_name'], final_module)
    else:
        fc_feature = getattr(model, Final_layer_info['Layer_name']).in_features
        final_module = nn.Linear(fc_feature, layer_num)
        setattr(model, Final_layer_info['Layer_name'], final_module)
    return model
