from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import pickle, h5py
import cv2
import torch
import torch
from torch.autograd import Variable
import json
from skimage.transform import resize
from skimage import img_as_bool
from PIL import Image
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
from torchvision import transforms


class NOISYACTIONSTEST(Dataset):
    def __init__(self, root = '', train=True, fold=1, transform=None, frames_path='', mode='single'):

        self.root = root
        self.frames_path = frames_path
        self.train = train
        self.fold = fold
        self.mode = mode
        # To get total length
        self.class_names = {}
        self.video_paths, self.targets, self.starts = self.build_paths()
        self.targets = np.array(self.targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, video_label, start = self.video_paths[idx], self.targets[idx], self.starts[idx]
        video = self.get_video(video_path, start)
        if self.mode == 'multi':
            video_label = torch.tensor(video_label)
            video_label = torch.zeros(6146).scatter_(0,video_label,1.)
        return video, video_label, video_path.replace(self.frames_path,'')

    def get_video(self, video_path, start):
        start_frame = start
        video_container = []
        frames_list = os.listdir(video_path)
        # The indexing of my frames start from 1, hence made appropriate changes
        for item in range(start_frame, start_frame + 16):
            #image_name = 'image_' + str(item).zfill(5) + '.jpg'
            image_name = frames_list[item]
            image_path = os.path.join(video_path, image_name)
            current_image = Image.open(image_path).convert('RGB')
            video_container.append(current_image)

        if self.transform is not None:
            clip = [self.transform(img) for img in video_container]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip


    def build_paths(self):
        data_paths = []
        targets = []
        startings = []
        if self.mode == 'single':
            if self.train:
                annotation_path = os.path.join(self.root, 'noisyActionsSingleTrainTestList', 'trainlist25k.txt')
            else:
                annotation_path = os.path.join(self.root, 'noisyActionsSingleTrainTestList', 'testlist25k.txt')
            class_ind_path = os.path.join(self.root, 'noisyActionsSingleTrainTestList', 'classInd25k.txt')
        elif self.mode == 'multi' or self.mode == 'partial':
            if self.train:
                annotation_path = os.path.join(self.root, 'noisyActionsMultiTrainTestList', 'trainlist25k.txt')
            else:
                annotation_path = os.path.join(self.root, 'noisyActionsMultiTrainTestList', 'annotTest100k.txt')
            class_ind_path = os.path.join(self.root, 'noisyActionsMultiTrainTestList', 'classInd100k.txt')
        
        class_mapping = {}
        with open(class_ind_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            for item in data:
                class_mapping[int(item[0])-1] = item[1]
        
        annotation_data = {}
        with open(annotation_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            if self.mode == 'single':
                for item in data:
                    keyName = item[0].replace('.avi','').replace('.mp4', '')
                    annotation_data[keyName] = int(item[1])-1
            elif self.mode == 'multi':
                for item in data:
                    #print(item)
                    keyName = item[0].replace('.avi','').replace('.mp4', '')
                    annotation_data[keyName] = [int(label)-1 for label in item[1].strip().rstrip().split('|')]
            
            if self.mode == 'multi':
                for i in annotation_data:
                    for j in annotation_data[i]:
                        self.class_names[j] = 1
            else:
                for i in annotation_data:
                    self.class_names[annotation_data[i]] = 1
#             for item in data:
#                 label_name = item[0].split('/')[0]
#                 annotation_data[item[0].replace('.avi','')] = class_mapping[label_name]

        if self.train:
            for key in annotation_data:
                video_path = os.path.join(self.frames_path, key) 
                n_frame = len(os.listdir(video_path))
                # The indexing of my frames start from 1, hence made appropriate changes
                start_frames = list(range(1, n_frame + 1, 16))
                while (n_frame-1) - start_frames[-1] < 16:
                    start_frames = start_frames[:-1]
                for item in start_frames:
                    data_paths.append(video_path)
                    targets.append(annotation_data[key])
                    startings.append(item)

        else:
            for key in annotation_data:
                try:
                    video_path = os.path.join(self.frames_path, key)
                    n_frame = len(os.listdir(video_path))
                    # The indexing of my frames start from 1, hence made appropriate changes
                    start_frames = list(range(1, n_frame, 16))
                    while (n_frame-1) - start_frames[-1] < 16:
                        start_frames = start_frames[:-1]
                    for item in start_frames:
                        data_paths.append(video_path)
                        targets.append(annotation_data[key])
                        startings.append(item)
                except Exception as e:
                    #print(video_path)
                    pass


        return data_paths, targets, startings

