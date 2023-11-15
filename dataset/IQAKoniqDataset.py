# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pandas as pd
import scipy.io
import numpy as np
import os
import cv2

from torchvision import transforms
from .BaseSRDataset import BaseSRDataset


class IQAKoniqDataset(BaseSRDataset):
    def __init__(self, img_folder, ann_file, scale=1, test_mode=False):
        super().__init__(scale, test_mode)
        self.img_folder = str(img_folder)
        self.ann_file = pd.read_csv(ann_file, error_bad_lines=True)
        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        if test_mode:
            self.data_infos = self.ann_file[self.ann_file.set == "test"].reset_index()
            self.gt_labels = self.ann_file[self.ann_file.set == "test"].MOS.values
        else:
            self.data_infos = self.ann_file[
                self.ann_file.set == "training"
            ].reset_index()
            self.gt_labels = self.ann_file[self.ann_file.set == "training"].MOS.values

    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        img = cv2.imread(osp.join(self.img_folder, self.data_infos["image_name"][idx]))
        img = self.train_transform(img)
        results = dict(
            img=img,
            gt=self.gt_labels[idx] / 100,
        )
        results["scale"] = self.scale
        return results


class IQALIVEITWDataset(BaseSRDataset):
    def __init__(self, img_folder, file_path, scale=1, test_mode=True):
        super().__init__(scale, test_mode)
        self.img_folder = str(img_folder)
        self.data_infos = scipy.io.loadmat(
            osp.join(file_path, "AllImages_release.mat")
        )["AllImages_release"][7:]
        self.gt_labels = scipy.io.loadmat(osp.join(file_path, "AllMOS_release.mat"))[
            "AllMOS_release"
        ][0][7:]
        # self.std_mat = scipy.io.loadmat(os.path.join(file_path, 'AllStdDev_release.mat'))

    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        results = dict(
            lq_path=osp.join(self.img_folder, self.data_infos[idx][0][0]),
            gt=self.gt_labels[idx] / 100,
        )
        results["scale"] = self.scale
        return results


class IQAAVADataset(BaseSRDataset):
    def __init__(self, img_folder, file_path, scale=1, test_mode=True):
        super().__init__(scale, test_mode)
        self.img_folder = str(img_folder)
        if test_mode:
            self.data_infos = np.loadtxt(
                os.path.join(file_path, "test_ava_name.txt"), dtype=int
            )
            self.gt_labels = np.loadtxt(
                os.path.join(file_path, "test_ava_score.txt"), dtype=float
            )[:, 0]
        else:
            self.data_infos = np.loadtxt(
                os.path.join(file_path, "train_ava_name.txt"), dtype=int
            )
            self.gt_labels = np.loadtxt(
                os.path.join(file_path, "train_ava_score.txt"), dtype=float
            )[:, 0]
        # self.std_mat = scipy.io.loadmat(os.path.join(file_path, 'AllStdDev_release.mat'))

    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        results = dict(
            lq_path=osp.join(self.img_folder, str(self.data_infos[idx]) + ".jpg"),
            gt=self.gt_labels[idx] / 100,
        )
        results["scale"] = self.scale
        return results
