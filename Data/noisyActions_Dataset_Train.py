from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
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


class NOISYACTIONSTRAIN(Dataset):
    def __init__(self, root = '', train=True, fold=1, transform=None, frames_path='', mode='single'):

        self.root = root
        self.frames_path = frames_path
        self.train = train
        self.fold = fold
        self.mode = mode
        # To get total length
        self.class_names = {}
        self.video_paths, self.targets = self.build_paths()
        #self.targets = np.array(self.targets)
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, video_label = self.video_paths[idx], self.targets[idx]
        video = self.get_video(video_path)
        # For Multi-Label
        if self.mode == 'multi' or self.mode == 'multi_mixup':
            video_label = torch.tensor(video_label)
            video_label = torch.zeros(len(self.class_names)).scatter_(0,video_label,1.)
        """elif self.mode == 'partial':
            video_label = torch.tensor(video_label['single'][0])
            partial_label = torch.tensor(video_label['partial'])
        video_label = torch.tensor(video_label)
        if self.mode != 'partial':
            return video, video_label
        else:
            return video, (video_label, partial_label)"""
        return video, video_label

    def get_video(self, video_path):
        no_frames = len(os.listdir(video_path))
        skip_rate = random.randint(1,5)
        total_frames = 16*skip_rate
        
        # Need to modify this to accomodate for random skip rates
        #infinite_counter = 0
        if no_frames >= 16:
            while total_frames > no_frames:
                skip_rate = skip_rate -1
                if skip_rate == 0:
                    skip_rate = 1
                total_frames = 16*skip_rate
                """infinite_counter += 1
                if infinite_counter > 1000:
                    print(video_path)"""
            allFrames = os.listdir(video_path)
        else:
            tempAllFrames = os.listdir(video_path)
            skip_rate = 1
            total_frames = skip_rate*16
            allFrames = []
            for i in tempAllFrames:
                allFrames.append(i)
                allFrames.append(i)
        # The indexing of my frames start from 1, hence made appropriate changes
        try:
            start_frame = random.randint(1, no_frames - total_frames) ## 32, 16 frames
        except:
            start_frame = 1
        video_container = []
        #allFrames = os.listdir(video_path)
        for item in range(start_frame, start_frame + total_frames, skip_rate):
            #image_name = 'image_' + str(item).zfill(5) + '.jpg'
            # Frames already skipped during decoding, hence doing this
            try:
                image_name = allFrames[item - 1]
            except:
                print(video_path, item, len(allFrames), start_frame, total_frames, skip_rate)
                raise
            image_path = os.path.join(video_path, image_name)
            current_image = Image.open(image_path).convert('RGB')
            video_container.append(current_image)

        if self.transform is not None:
            self.transform.randomize_parameters()
            clip = [transforms.functional.normalize(self.transform(img), normal_mean, normal_std) for img in video_container]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip


    def build_paths(self):
        data_paths = []
        targets = []
        if self.mode == 'single':
            if self.train:
                annotation_path = os.path.join(self.root, 'noisyActionsSingleTrainTestList', 'trainlist25k.txt')
            else:
                annotation_path = os.path.join(self.root, 'noisyActionsSingleTrainTestList', 'testlist25k.txt')
        elif self.mode == 'multi' or self.mode == 'partial':
            if self.train:
                annotation_path = os.path.join(self.root, 'noisyActionsMultiTrainTestList', 'trainlist100k.txt')
            else:
                annotation_path = os.path.join(self.root, 'noisyActionsMultiTrainTestList', 'testlist100k.txt')
        
        annotation_data = {}
        with open(annotation_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            if self.mode == 'single':
                for item in data:
                    keyName = item[0].replace('.avi','').replace('.mp4', '')
                    annotation_data[keyName] = int(item[1])-1
            elif self.mode == 'multi' or self.mode == 'partial':
                for item in data:
                    keyName = item[0].replace('.avi','').replace('.mp4', '')
                    annotation_data[keyName] = [int(label)-1 for label in item[1].strip().rstrip().split('|')]
        
        # Will use the class names dict for calculating mean number of classes as well
        if self.mode == 'multi' or self.mode == 'partial':
            for i in annotation_data:
                for j in annotation_data[i]:
                    if j not in self.class_names:
                        self.class_names[j] = 0
                    self.class_names[j] += 1
        elif self.mode == 'single':
            for i in annotation_data:
                self.class_names[annotation_data[i]] = 1
        
        if self.mode in ['single', 'multi']:
            for key in annotation_data:
                data_paths.append(os.path.join(self.frames_path, key)) 
                targets.append(annotation_data[key])
        elif self.mode == 'partial':
            # Get Average number of videos per class
            threshold = sum([self.class_names[i] for i in self.class_names])/len(self.class_names)
            oldClassDistribution = {}
            for i in annotation_data:
                for j in annotation_data[i]:
                    if j not in oldClassDistribution:
                        oldClassDistribution[j] = []
                    oldClassDistribution[j].append(i)
            for i in oldClassDistribution:
                random.shuffle(oldClassDistribution[i])
            newVideo_ClassDistribution = {}
            labelCount = {i:0 for i in oldClassDistribution}
            for i in oldClassDistribution:
                for j in oldClassDistribution[i]:
                    if j not in newVideo_ClassDistribution:
                        newVideo_ClassDistribution[j] = {'single':[], 'partial':[]}
                    if labelCount[i] <= threshold:
                        if newVideo_ClassDistribution[j]['single'] == []:
                            newVideo_ClassDistribution[j]['single'].append(i)
                            labelCount[i] += 1
                        else:
                            newVideo_ClassDistribution[j]['partial'].append(i)
                    else:
                        newVideo_ClassDistribution[j]['partial'].append(i)
            for i in newVideo_ClassDistribution:
                if newVideo_ClassDistribution[i]['single'] == []:
                    randomClass = random.sample(annotation_data[i], 1)[0]
                    newVideo_ClassDistribution[i]['single'].append(randomClass)
                    newVideo_ClassDistribution[i]['partial'] = list(set(newVideo_ClassDistribution[i]['partial']).difference(set(newVideo_ClassDistribution[i]['single'])))
            for key in newVideo_ClassDistribution:
                data_paths.append(os.path.join(self.frames_path, key))
                target = torch.zeros(len(self.class_names)).scatter_(0, torch.tensor(newVideo_ClassDistribution[key]['partial']).long(), -1.).scatter_(0, torch.tensor(newVideo_ClassDistribution[key]['single']).long(), 1.)
                targets.append(target)

        return data_paths, targets
