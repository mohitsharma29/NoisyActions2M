import math
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from art.classifiers import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.defences.trainer import AdversarialTrainerFBFPyTorch
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent,FastGradientMethod

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class Robust(object):
    def __init__(self,model,no_class):
        self.model=model
        criterion= nn.CrossEntropyLoss()

        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.005)

        self.classifier = PyTorchClassifier(
        model=self.model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3,16,112,112),
        nb_classes=no_class)
        self.attack = FastGradientMethod(estimator=self.classifier, eps=0.2)
    def __call__(self, pic):
            """
            Args:
                pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            Returns:
                Tensor: Converted image.
            """
            #pic=torch.from_numpy(pic).float()
            pic =self.attack.generate(pic.cpu())
            return torch.from_numpy(pic).float()