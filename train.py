from math import sqrt
import os
import json
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch
import torch.nn as nn
from torch.nn import functional as F

# import torchvision
import torch.optim
import argparse

from typing import Union, Tuple

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.IQAKoniqDataset import IQAKoniqDataset

from clipiqa_arch import CLIPIQA
from CLIP.model import CLIP
from CLIP.model import Transformer
import scipy
from scipy import stats
import pandas as pd

import numpy as np

from test_function import inference

import clip_score
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import clip

import pyiqa
import shutil

task_name = "train0"


def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    plcc_result = stats.pearsonr(x, y)[0]
    return np.round(plcc_result, 3)


def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    srocc_result = plcc(xranks, yranks)
    return srocc_result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def random_crop(img):
    b, c, h, w = img.shape
    hs = random.randint(0, h - 224)
    hw = random.randint(0, w - 224)
    return img[:, :, hs : hs + 224, hw : hw + 224]


def train(config):
    task_name = "train0"

    writer = SummaryWriter("./" + task_name + "/" + "tensorboard_" + task_name)

    dstpath = "./" + task_name + "/" + "train_scripts"
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    shutil.copy("train.py", dstpath)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_id}")
    else:
        device = torch.device("cpu")
    print(device)
    # load clip
    model, preprocess = clip.load(
        "RN50", device=torch.device("cpu"), download_root="./clip_model/"
    )  # ViT-B/32 RN50
    model.to(device)

    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # load model
    model = CLIPIQA(model_type="clipiqa", backbone="RN50", pretrained=True)
    model.to(device=device)
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device)
    model.load_state_dict(state_dict)

    # add pretrained model weights
    model.apply(weights_init)
    if config.load_pretrain == True:
        print(
            "The load_pretrain is True, thus num_reconstruction_iters is automatically set to 0."
        )
        config.num_reconstruction_iters = 0
        state_dict = torch.load(config.pretrain_dir)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # U_net.load_state_dict(torch.load(config.pretrain_dir))
        torch.save(
            model.state_dict(),
            config.train_snapshots_folder + "pretrained_network" + ".pth",
        )
    else:
        if config.num_reconstruction_iters < 200:
            print(
                "WARNING: For training from scratch, num_reconstruction_iters should not lower than 200 iterations!\nAutomatically reset num_reconstruction_iters to 1000 iterations..."
            )
            config.num_reconstruction_iters = 1000

    # load dataset
    train_dataset = IQAKoniqDataset(config.images_path, config.ann_file)  # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # loss
    criterion = torch.nn.MSELoss()

    # load gradient update strategy.
    params_to_update = []
    print("making optimizer")
    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            params_to_update.append(param)

    optimizer = torch.optim.SGD(
        params=params_to_update, lr=config.train_lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    # initial parameters
    model.train()
    total_iteration = 0

    # Start training!
    outcome_list = []
    for epoch in range(config.num_epochs):
        pred_epoch = []
        labels_epoch = []
        sumLoss = 0
        for data in tqdm(train_loader):
            # print(data)
            img = data["img"]
            probs = model(img.to(device))

            # 梯度回传 grad backwards
            optimizer.zero_grad()
            gt = data["gt"].reshape(data["gt"].size(), 1)
            loss = criterion(probs, gt.to(device))

            loss.requires_grad = True
            loss.backward()
            sumLoss += loss
            optimizer.step()
            scheduler.step()
            pred_batch_numpy = probs.data.cpu().numpy()
            labels_batch_numpy = gt.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        p_srocc = srocc(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        p_plcc = plcc(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        print(
            "train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}".format(
                epoch, sumLoss, p_srocc, p_plcc
            )
        )
        outcome_list.append(
            {
                "train epoch": epoch,
                "loss": sumLoss.data.cpu().numpy().tolist(),
                "SRCC": p_srocc,
                "PLCC": p_plcc,
            }
        )
    file_name = str(config.gpu_id) + "_" + str(config.num_epochs) + "trainLog.json"

    # 将列表保存为 JSON 格式的文件
    with open(file_name, "w") as file:
        json.dump(outcome_list, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument(
        "-r",
        "--images_path",
        type=str,
        default="/app/users/liz/experiment/CLIPIQA/koniq10k_1024x768/1024x768",
    )
    parser.add_argument(
        "-a",
        "--ann_file",
        type=str,
        default="/app/users/liz/experiment/CLIPIQA/CLIP-IQA-2-3.8/koniq10k_scores_and_distributions/koniq10k_distributions_sets.csv",
    )
    parser.add_argument("--length_prompt", type=int, default=16)
    parser.add_argument("--thre_train", type=float, default=90)
    parser.add_argument("--thre_prompt", type=float, default=60)
    parser.add_argument(
        "--train_lr", type=float, default=0.02
    )  # 0.00002#0.00005#0.0001
    parser.add_argument("-g", "--gpu_id", type=int, default=0)
    parser.add_argument("--T_max", type=float, default=100)
    parser.add_argument("--eta_min", type=float, default=1e-6)  # 1e-6
    parser.add_argument("--weight_decay", type=float, default=0.001)  # 0.0001
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=2000)  # 3000
    parser.add_argument("--num_reconstruction_iters", type=int, default=0)  # 1000
    parser.add_argument("--num_clip_pretrained_iters", type=int, default=0)  # 8000
    parser.add_argument("--noTV_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--prompt_batch_size", type=int, default=16)  # 32
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--display_iter", type=int, default=20)
    parser.add_argument("--snapshot_iter", type=int, default=20)
    parser.add_argument("--prompt_display_iter", type=int, default=20)
    parser.add_argument("--prompt_snapshot_iter", type=int, default=100)
    parser.add_argument(
        "--train_snapshots_folder",
        type=str,
        default="./" + task_name + "/" + "snapshots_train_" + task_name + "/",
    )
    parser.add_argument(
        "--prompt_snapshots_folder",
        type=str,
        default="./" + task_name + "/" + "snapshots_prompt_" + task_name + "/",
    )
    parser.add_argument(
        "--load_pretrain", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default="./pretrained_models/init_pretrained_models/init_enhancement_model.pth",
    )
    parser.add_argument(
        "--load_pretrain_prompt",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
    )
    parser.add_argument(
        "--prompt_pretrain_dir",
        type=str,
        default="./pretrained_models/init_pretrained_models/init_prompt_pair.pth",
    )

    config = parser.parse_args()

    train(config)
