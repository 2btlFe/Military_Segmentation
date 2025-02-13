import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import scipy
import pickle
import os
import addict
import json



def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cityscale_data_partition():
    # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        if x % 10 < 8 :
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def spacenet_data_partition():
    # dataset partition
    with open('./spacenet/data_split.json','r') as jf:
        data_list = json.load(jf)
        # data_list = data_list['test'] + data_list['validation'] + data_list['train']
    # train_list = [tile_index for _, tile_index in data_list['train']]
    # val_list = [tile_index for _, tile_index in data_list['validation']]
    # test_list = [tile_index for _, tile_index in data_list['test']]
    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']
    return train_list, val_list, test_list


def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    # import ipdb
    # ipdb.set_trace()
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    
    # 이걸 왜 64 ~ 448로 하지?
    
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info

