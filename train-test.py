import argparse
import configparser
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle 
import scipy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.functional import block_diag
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Data.UCF101 import get_ucf101
from Data.HMDB51 import get_hmdb51
from Data.NOISYACTIONS import get_noisyActions
from utils import AverageMeter, accuracy
from transforms.mixup import MixupTransform
import torch.nn as nn
import torchmetrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Remove later: Ignoring warnings for multi label torch.tensor() casting in noisy actions train loader(mainly)
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

DATASET_GETTERS = {'ucf101': get_ucf101, 'hmdb51': get_hmdb51, 'noisy_actions': get_noisyActions}


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

def save_checkpoint(state, is_best, checkpoint, epoch):
    filename=f'checkpoint_{epoch}.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main_training_testing(EXP_NAME):
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--out', default=f'results/{EXP_NAME}', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--arch', default='resnet3D18', type=str,
                        help='dataset name')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--val-frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')
    parser.add_argument('--pretrain-path', default=None, type=str,
                        help='path of checkpoint to finetune')
    parser.add_argument('--n-finetune-classes', default=101, type=int,
                        help='Fine tune networks with this class number')
    parser.add_argument('--training-mode', default='single', type=str,
                        help='Type of training, single label or multi label')
    parser.add_argument('--multiLabelLoss', default='bce', type=str,
                        help='Loss to use with Multi Label Training')
    parser.add_argument('--mp', default=1, type=int, help='Enable MP using 1')
    parser.add_argument('--testAndSaveEvery', default=5, type=int, help='Test and save the model at these many epochs')
    parser.add_argument('--no-val', default=False, type=bool, help='Whether to validate or not')
    parser.add_argument('--pretext-ssl', default=False, type=bool, help='Whether you are using Pretext SSL')
    parser.add_argument('--adv-attack', default=False, type=bool, help='Adv. Attack on Test set')
    parser.add_argument('--noise-corr', default=False, type=bool, help='Add noise correction')
    parser.add_argument('--noise-corr_type', default='mixup', type=str, 
                        help='Type of noise correction you want to use')
    args = parser.parse_args()
    best_acc = 0
    #best_acc_2 = 0
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    args.gpu_id = 0

    def create_model(args):
        if args.arch == 'resnet3D18':
            from models import resnet
            model = resnet.resnet18(
                num_classes=args.num_class,
                shortcut_type="B",
                sample_size=112,
                sample_duration=16,
                ssl=False)
        elif args.arch == 'r2plus1d_18':
            from models.resnet2p1d import generate_model as gm
            model = gm(n_classes=args.num_class, model_depth=18)
        if args.pretrain_path is not None:
            from torch import nn
            # Uncomment below line only for old ICME checkpoints
            #model = nn.DataParallel(model, device_ids=None)
            print('loading pretrained model {}'.format(args.pretrain_path))
            pretrain = torch.load(args.pretrain_path, map_location='cuda:0')

            if not args.pretext_ssl:
                model.load_state_dict(pretrain['state_dict'])
            else:
                def load_pretrained_weights(ckpt_path):
                    """load pretrained weights and adjust params name."""
                    adjusted_weights = {}
                    pretrained_weights = torch.load(ckpt_path, map_location='cuda:0')
                    for name, params in pretrained_weights.items():
                        if 'base_network' in name:
                            name = name[name.find('.')+1:]
                            adjusted_weights[name] = params
                            #print('Pretrained weight name: [{}]'.format(name))
                    return adjusted_weights
                model.load_state_dict(load_pretrained_weights(args.pretrain_path), strict=False)
            # Uncomment for non ICME checkpoints
            #model.module.fc = nn.Linear(model.module.fc.in_features,
            #                            args.n_finetune_classes)
            #model.module.fc = model.module.fc.cuda()
            model.fc = nn.Linear(model.fc.in_features,
                                        args.n_finetune_classes)
            model.fc = model.fc.cuda()
        return model
    
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)
    
    # Adding MP Loss scaler (https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam)
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.mp == 1 else False)
    
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    model = create_model(args)    
    model.to(args.device)

    train_dataset, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.train_frames_path, args.val_frames_path,args.training_mode)
   
    args.iteration = len(train_dataset) // args.batch_size // args.world_size
    train_sampler = RandomSampler 
    #train_sampler = SequentialSampler
    if args.noise-corr:
        config = configparser.ConfigParser()
        if args.noise-corr_type=="mixup":
            print("Reading mixup config from: transforms/transforms_config.ini")
            config.read("transforms/transforms_config.ini")
            mixup_transform = MixupTransform(
                config["alpha"],
                num_classes=config["num_classes"],
                cutmix_alpha=config["cutmix_alpha"],
                cutmix_minmax=config["cutmix_minmax"],
                mix_prob=config["mix_prob"],
                switch_prob=config["switch_prob"],
                mode=config["mode"],
                label_smoothing=config["label_smoothing"],
            )
            train_dataset = mixup_transform(train_dataset)
        elif args.noise-corr_type=="nested_dropout":
            continue
        else:
            print("Invalid noise correction type, stopping training....")
            return
    else:
        print("Continuing without noise correction....")
        continue
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.iteration, args.total_steps)

    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = start_epoch
    
    test_accs = []
    model.zero_grad()
    best_acc = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
     
        train_loss = train(args, train_loader, model, optimizer, scheduler, epoch, scaler)
        #train_loss = 0.0
        test_loss = 0.0
        test_acc_2 = 0.0
        test_model = model

        if (epoch + 1)%args.testAndSaveEvery == 0:
            writer.add_scalar('train/1.train_loss', train_loss, epoch)
            if args.no_val == False:
                test_loss, test_acc = test(args, test_loader, test_model, epoch)
                if args.training_mode == 'single':
                    writer.add_scalar('test/1.test_acc', test_acc, epoch)
                    writer.add_scalar('test/1.test_loss', test_loss, epoch)
                elif args.training_mode == 'multi':
                    writer.add_scaler('test/1.test_acc', test_acc[0], epoch)
                    writer.add_scaler('test/1.test_f1', test_acc[1], epoch)
                    writer.add_scaler('test/1.test_hamming', test_acc[2], epoch)
                    writer.add_scaler('test/1.test_jaccard', test_acc[3], epoch)
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
            else:
                is_best = False
                best_acc = 0.0
                test_acc = 0.0

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }, is_best, args.out, epoch)
            test_accs.append(test_acc)
    with open(f'{args.out}/score_logger.txt', 'a+') as ofile:
        ofile.write(f'Last Acc (after softmax): {test_acc}, Best Acc (after softmax): {best_acc}\n')

    if args.local_rank in [-1, 0]:
        writer.close()

def train(args, labeled_trainloader, model, optimizer, scheduler, epoch, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if args.training_mode == 'single':
        criterion = nn.CrossEntropyLoss()
    elif args.training_mode == 'multi':
        if args.multiLabelLoss == 'bce':
            print('Using BCE')
            criterion = nn.BCEWithLogitsLoss()
        elif args.multiLabelLoss == 'focal':
            from loss_functions import focal_loss
            criterion = focal_loss.WeightedFocalLoss()
            print('Using Focal')
        elif args.multiLabelLoss == 'asym':
            from loss_functions import asym_loss
            # https://github.com/Alibaba-MIIL/ASL/issues/22
            # Losses are high with ASL, because of sum reduction, instead of mean
            #criterion = asym_loss.AsymmetricLossOptimized(gamma_neg=0, gamma_pos=0, clip=0) -> BCE
            print('Using Asym')
            criterion = asym_loss.AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05,disable_torch_grad_focal_loss=True)
        elif args.multiLabelLoss == 'balanced':
            print('Using Balanced')
            from loss_functions.dist_bal_loss import resample_loss
            from loss_functions.dist_bal_loss.class_freq import save_class_freq
            save_class_freq(args)
            criterion = resample_loss.ResampleLoss(reweight_func='rebalance')
        else:
            assert 2 == 1
    elif args.training_mode == 'partial':
        train_sampler = RandomSampler
        train_dataset, _ = DATASET_GETTERS[args.dataset]('Data', args.train_frames_path, args.val_frames_path, args.training_mode)
        labeled_trainloader = DataLoader(
            train_dataset,
            sampler=train_sampler(train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False)
        criterion = nn.CrossEntropyLoss()
    else:
        print('Invalid training mode')
        assert 2 == 1

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = labeled_trainloader
    model.train()
    
    for batch_idx, (inputs_x, targets_x) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs_x.to(args.device)
        """if args.training_mode == 'partial':
            partial_x = deepcopy(targets_x[1])
            targets_x = targets_x[0]"""
        targets_x = targets_x.to(args.device)
        # MP loop
        # For Asymmetric Multi-Label Loss its recommended to not use MP
        if args.training_mode == 'multi' and args.multiLabelLoss == 'asym':
            logits_x = model(inputs)
            loss = criterion(logits_x, targets_x)
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            with torch.cuda.amp.autocast(enabled=True if args.mp == 1 else False):
                logits_x = model(inputs)
                if args.training_mode != 'partial':
                    loss = criterion(logits_x, targets_x)
                else:
                    logits_x[targets_x == -1.0] = float('-inf')
                    targets_x[targets_x == -1.0] = 0.0
                    targets_x = targets_x.long()
                    loss = criterion(logits_x, torch.max(targets_x, 1)[1])
            
            scaler.scale(loss).backward()
            losses.update(loss.item())
            
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg


def test(args, test_loader, model, epoch, training_mode='single'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)
    
    if args.adv_attack == True:
        from robust_transform import Robust
        if args.pretrain_path is not None:
            adv_attack_model = Robust(model, args.n_finetune_classes)
        else:
            adv_attack_model = Robust(model, args.num_class)
    
    for batch_idx, (inputs, targets, video_name) in enumerate(test_loader):
            data_time.update(time.time() - end)
            if args.adv_attack == True:
                inputs = adv_attack_model(inputs)
            with torch.no_grad():
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs)
                if args.training_mode == 'single':
                    loss = F.cross_entropy(outputs, targets)
                    out_prob = F.softmax(outputs, dim=1)
                elif args.training_mode == 'multi':
                    if args.multiLabelLoss == 'bce':
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(outputs, targets)
                    elif args.multiLabelLoss == 'focal':
                        from loss_functions import focal_loss
                        criterion = focal_loss.WeightedFocalLoss()
                        loss = criterion(outputs, targets)
                    elif args.multiLabelLoss == 'asym':
                        from loss_functions import asym_loss
                        criterion = asym_loss.AsymmetricLossOptimized()
                        loss = criterion(outputs, targets)
                    elif args.multiLabelLoss == 'balanced':
                        from loss_functions.dist_bal_loss import resample_loss
                        criterion = resample_loss.ResampleLoss()
                        loss = criterion(outputs, targets)
                    out_prob = torch.sigmoid(outputs)
                    out_prob[out_prob >= 0.5] = 1
                elif args.training_mode == 'partial':
                    pass
                else:
                    assert 2 == 1
                out_prob = out_prob.cpu().numpy().tolist()
                targets = targets.cpu().numpy().tolist()
                #outputs = outputs.cpu().numpy().tolist()

                for iterator in range(len(video_name)):
                    if video_name[iterator] not in predicted_target:
                        predicted_target[video_name[iterator]] = []

                    """if video_name[iterator] not in predicted_target_not_softmax:
                        predicted_target_not_softmax[video_name[iterator]] = []"""

                    if video_name[iterator] not in ground_truth_target:
                        ground_truth_target[video_name[iterator]] = []

                    predicted_target[video_name[iterator]].append(out_prob[iterator])
                    #predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                    ground_truth_target[video_name[iterator]].append(targets[iterator])

                losses.update(loss.item(), inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                if not args.no_progress:
                    test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg
                    ))
    
    model.train()
    if not args.no_progress:
        test_loader.close()
    
    if args.training_mode == 'single':
        for key in predicted_target:
            clip_values = np.array(predicted_target[key]).mean(axis=0)
            video_pred = np.argmax(clip_values)
            predicted_target[key] = video_pred
        
        for key in ground_truth_target:
            clip_values = np.array(ground_truth_target[key]).mean(axis=0)
            ground_truth_target[key] = int(clip_values)

        pred_values = []
        target_values = []

        for key in predicted_target:
            pred_values.append(predicted_target[key])
            target_values.append(ground_truth_target[key])
        
        pred_values = np.array(pred_values)
        target_values = np.array(target_values)

        secondary_accuracy = (pred_values == target_values)*1
        # Removed Multiplication by 100
        secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))
        print(f'test accuracy: {secondary_accuracy}')
        return losses.avg, secondary_accuracy
    elif args.training_mode == 'multi':
        for key in predicted_target:
            clip_values = np.array(predicted_target[key]).mean(axis=0)
            predicted_target[key] = clip_values
        for key in ground_truth_target:
            clip_values = np.array(ground_truth_target[key]).mean(axis=0)
            ground_truth_target[key] = int(clip_values)
        pred_values = []
        target_values = []
        for key in predicted_target:
            pred_values.append(predicted_target[key])
            target_values.append(ground_truth_target[key])
        pred_values = np.array(pred_values)
        target_values = np.array(target_values)
        accuracy = torchmetrics.functional.accuracy(pred_values, target_values)
        f1_score = torchmetrics.functional.f1(pred_values, target_values)
        hamming = torchmetrics.functional.hamming_distance(pred_values, target_values)
        jaccard = torchmetrics.functional.iou(pred_values, target_values)
        print(f'Accuracy {accuracy}, F1 score {f1_score}, Hamming Score {hamming}, Jaccard Score {jaccard}')
        return losses.avg, (accuracy, f1_score, hamming, jaccard)

if __name__ == '__main__':
    cudnn.benchmark = True
    EXP_NAME = 'UCF101_SUPERVISED_TRAINING'
    main_training_testing(EXP_NAME)
