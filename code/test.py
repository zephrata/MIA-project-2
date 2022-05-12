# -*- coding: utf-8 -*-
import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import nibabel as nib
from torch import optim
from datetime import datetime
from Resnet3D import resnet10,resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
import pandas as pd


from torch.utils.data import Dataset, DataLoader


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_num', help="Input the amount of GPU you need", default=0, type=int)
    parser.add_argument('--GPU_no', help="Input the No of GPU you want", default='', type=str)
    parser.add_argument('--batch_size', help="Input the batch number", default=2, type=int)
    parser.add_argument('--epoch', help="Input the number of epoch number", default=40, type=int)
    parser.add_argument('--pretrain', help="Input if you need a pre trained model", default=False, type=bool)
    parser.add_argument('--model_depth', help="Input your resnet depth", default=10, type=int)
    parser.add_argument('--lr', help='Input learning rate', default=0.0001, type=float)
    parser.add_argument('--norm', help='Input your normalisation method', default='../undelineated/volumes', type=str)
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, dir_name, label_name, setting):
        self.dir_name = dir_name
        self.setting = setting
        if label_name is not None:
            self.y = pd.DataFrame(pd.read_csv(label_name))['Age']
        else:
            self.y = None
        # self.y = np.load(os.path.join(dir_name, '001_T1.npy'))

    def __getitem__(self, index):
        if self.setting == 'val':
            #self.agent = nib.load(os.path.join(self.dir_name, str(index+502).zfill(3)+'_T1.nii.gz'))
            self.agent = nib.load(os.path.join(self.dir_name, str(index+512).zfill(3)+'_T1.nii.gz'))
        else:
            self.agent = nib.load(os.path.join(self.dir_name, str(index+1).zfill(3)+'_T1.nii.gz'))
        data = np.array(self.agent.dataobj)
        #data = data[np.newaxis, 32:-32, 32:-32, 14:-50]
        data = data[np.newaxis, :, :, :]
        if self.y is not None:
            return torch.from_numpy(data.astype('float32')), torch.tensor(float(self.y[index]))
        else:
            return torch.from_numpy(data.astype('float32')), torch.tensor(0.)

    def __len__(self):
        if self.y is not None:
            return self.y.shape[0]
        else:
            return len(os.listdir(self.dir_name))


if __name__ == "__main__":
    args = init()
    para_name = '-'.join([key + ':' + str(value) for key, value in args.__dict__.items() if key != 'norm'])

    epoch = 30

    assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    model = eval('resnet'+str(args.model_depth)+'()').to(device)

    device = torch.device('cpu')
    model_name = 'output/model' + para_name + '_' + str(epoch) + '.pkl'
    old_model = torch.load(model_name, map_location=None if torch.cuda.is_available() else 'cpu')
    old_model_dict = {key.replace('module.', ''): value for key, value in old_model.state_dict().items()}
    model_dict = {key: value for key, value in old_model_dict.items() if key in model.state_dict().keys()}
    model.load_state_dict(model_dict)
    model.to(device)

    if device.type == 'cuda' and args.GPU_num > 1:
        if args.GPU_no:
            assert len(args.GPU_no) == args.GPU_num
            model = nn.DataParallel(model, [int(each) for each in args.GPU_no])
        else:
            model = nn.DataParallel(model, list(range(args.GPU_num)))

    
    #test_set = MyDataset('../final_eval/age_eval/skull_stripping/', '../final_eval/age_eval/ages_gt.csv', 'val')
    test_set = MyDataset('../final_test/age_eval/skull_stripping/', None, 'val')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_num = len(test_loader)
    
    model.eval()
    predict_list = []
    true_list = []
    test_loss = 0
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            outputs = model(inputs)
            predict_list.extend(list(outputs.detach().cpu().squeeze(0).numpy()))
            true_list.extend(list(labels.detach().cpu().squeeze(0).numpy()[0:]))

            test_loss += torch.sum(torch.abs(labels - outputs))
        for i in range(len(true_list)):
            if type(true_list[i]) is np.float32:
                true_list[i] = true_list[i]
            else:
                true_list[i] = true_list[i][0]
        for i in range(len(predict_list)):
            if type(predict_list[i]) is np.float32:
                predict_list[i] = predict_list[i]
            else:
                predict_list[i] = predict_list[i][0]
        #predict_list = [i[0] for i in predict_list]
        print(true_list, predict_list)
        print('Epoch %d Test loss %4.2f' % (epoch, test_loss/test_num))

    f = open('../final_test/age_eval/ages_gt.csv','w')
    f.write('ID,Age\n')
    for i in range(len(predict_list)):
        f.write(str(i+512) + ',' + str(predict_list[i]) + '\n')

