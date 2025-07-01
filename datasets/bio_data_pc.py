import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm
import copy
from typing import Union, Tuple
import h5py


class SMLMDataset(Dataset):
    def __init__(self, cfg, split, fast_dev_run=False, input_dim=3):
        assert split in ["train", "val", "test", "test-exp"], "split error value!"

        self.dataset_name = cfg.dataset_name
        self.tr_sample_size = cfg.tr_max_sample_points
        self.te_sample_size = cfg.te_max_sample_points
        self.scale = cfg.dataset_scale
        self.is_scale_z = cfg.is_scale_z
        self.is_random_sample = cfg.is_random_sample
        self.transforms = cfg.transforms_params if cfg.transforms else None
        self.noise_points_ratio = cfg.noise_points_ratio
        self.dataroot = cfg.data_dir
        self.split = split
        self.fast_dev_run = fast_dev_run
        self.input_dim = input_dim

        # if self.split != "train":
        #     self.is_random_sample = False

        data_split = {
            "local": {
                "train": [0, 5],
                "val": [1000, 1005],
                "test": [1001, 1002],
                "test-exp": [0, 1],
            },
            "remote": {
                "train": [0, 1000],
                "val": [1000, 1024],
                "test": [1001, 1002],
                "test-exp": [0, 1],
            },
        }
        if self.split != "test-exp":
            h5_file_path = os.path.join(self.dataroot, self.dataset_name)
        else:
            h5_file_path = os.path.join("datasets/region_x0_y0_z1_2048_16384_norm.h5")

        if self.fast_dev_run:
            start, end = data_split["local"][self.split]
        else:
            start, end = data_split["remote"][self.split]

        with h5py.File(h5_file_path, "r") as f:
            self.input_data = f["input_data"][start:end].astype(np.float32)
            self.gt_data = f["gt_data"][start:end].astype(np.float32)
            # self.original_data = f["original_data"][start:end].astype(np.float32)
            normalize_params = f["norm_params"]
            self.normalize_params = {
                "centroid": normalize_params["centroid"][start:end].astype(np.float32),
                "scale": normalize_params["scale"][start:end].astype(np.float32),
            }
        assert self.input_data.shape[0] == self.gt_data.shape[0]

        if self.is_scale_z:
            self.input_data, self.gt_data, self.normalize_params = SMLMDataset.scale_z(  # type: ignore
                self.input_data, self.gt_data, self.normalize_params
            )

        # if self.is_scale_half:
        #     # from -1~1 to -0.5~0.5
        #     self.input_data /= 2
        #     self.gt_data /= 2
        #     # self.original_data /= 2

        self.input_data = self.input_data * self.scale
        self.gt_data = self.gt_data * self.scale
        # self.original_data = self.original_data * self.scale

        print("input_data.shape:", self.input_data.shape)
        print("gt_data.shape:", self.gt_data.shape)
        # print("original_data.shape:", self.original_data.shape)
        print("local:", self.fast_dev_run)
        print("is_scale_z:", self.is_scale_z)
        print("scale:", self.scale)

        self.transforms = None
        # if self.transforms:
        #     self.transforms = self._get_transforms(cfg, self.split)

        if self.split == "train" and self.noise_points_ratio > 0:
            self.input_data = self._get_noise(self.input_data, self.noise_points_ratio)

        if self.noise_points_ratio > 0:
            print("noise added. input_data.shape:", self.input_data.shape)

        print("max of input_data:", np.max(self.input_data, axis=(0, 1)))
        print("min of input_data:", np.min(self.input_data, axis=(0, 1)))

        # ---

        print("Total number of data:%d" % len(self.input_data))
        print(
            "Min number of points: (train)%d (test)%d"
            % (self.tr_sample_size, self.te_sample_size)
        )

    # def _get_transforms(self, cfg, split):
    #     if split != "train":
    #         return None
    #     return utils.data_transforms.Compose(cfg.TRAIN.transforms_params)

    def _get_noise(self, pc, noise_points_ratio):
        """
        为点云添加噪声点

        Args:
            pc: 点云数据，形状为(点云数，点数，3)
            noise_points_ratio: 噪声点比例

        Returns:
            添加了噪声点的点云
        """
        if noise_points_ratio <= 0:
            return pc

        # 获取点云形状
        batch_size, num_points, _ = pc.shape

        # 计算要添加的噪声点数量
        noise_points_count = int(num_points * noise_points_ratio)

        # 如果噪声点数量为0，直接返回原始点云
        if noise_points_count == 0:
            return pc

        # 创建结果点云的副本
        result_pc = copy.deepcopy(pc)

        for i in range(batch_size):
            # 为每个点云生成随机噪声点
            # 分别按照点云x,y,z轴的最大最小值来设置噪声点的范围
            pc_min = np.min(pc[i], axis=0)  # 获取x,y,z三个维度的最小值
            pc_max = np.max(pc[i], axis=0)  # 获取x,y,z三个维度的最大值

            # 在每个维度的范围内分别随机生成噪声点坐标
            noise_points = np.zeros((noise_points_count, 3))
            for dim in range(3):  # 分别处理x,y,z三个维度
                noise_points[:, dim] = np.random.uniform(
                    low=pc_min[dim], high=pc_max[dim], size=noise_points_count
                )

            # 随机选择要删除的点的索引
            indices_to_remove = np.random.choice(
                num_points, noise_points_count, replace=False
            )

            # 删除选定的点
            remaining_indices = np.setdiff1d(np.arange(num_points), indices_to_remove)

            # 将剩余点与噪声点合并
            result_pc[i] = np.vstack([result_pc[i][remaining_indices], noise_points])

        return result_pc

    def __getitem__(self, index):
        result = {}
        partial_pc = copy.deepcopy(self.input_data[index])
        complete_pc = copy.deepcopy(self.gt_data[index])
        # original_pc = copy.deepcopy(self.original_data[index])
        if self.split == "train" or self.split == "val":
            if self.is_random_sample:
                tr_out = self.random_sample(complete_pc, self.tr_sample_size)
                te_out = self.random_sample(complete_pc, self.te_sample_size)
            else:
                tr_out = complete_pc[: self.tr_sample_size]
                te_out = complete_pc[
                    self.tr_sample_size : self.tr_sample_size + self.te_sample_size
                ]

            result["tr_points"] = torch.from_numpy(tr_out).float()
            result["te_points"] = torch.from_numpy(te_out).float()
            result["complete_pc"] = torch.from_numpy(complete_pc).float()

            # result['original_cloud'] = original_pc
            if self.split == "train":
                # augment
                if self.transforms is not None:
                    result = self.transforms(result)
        else:
            tr_out = complete_pc[: self.tr_sample_size]
            te_out = complete_pc[
                self.tr_sample_size : self.tr_sample_size + self.te_sample_size
            ]
            result["tr_points"] = torch.from_numpy(tr_out).float()
            result["te_points"] = torch.from_numpy(te_out).float()
            result["complete_pc"] = torch.from_numpy(complete_pc).float()
            normalize_params = {
                "centroid": self.normalize_params["centroid"][index],
                "scale": self.normalize_params["scale"][index],
            }
            result["normalize_params"] = normalize_params

        result["idx"] = index
        result["mean"] = torch.tensor(0)
        result["std"] = torch.tensor(1)
        
        # modify
        result["train_points"] = result.pop("tr_points")
        result["test_points"] = result.pop("te_points")
        result["cate_idx"] = torch.tensor(0)
        result["sid"] = torch.tensor(0)
        result["mid"] = torch.tensor(0)
        

        return result

    def __len__(self):
        return len(self.input_data)

    @staticmethod
    def scale_z(
        input_data, gt_data, params
    ) -> Union[Tuple[np.ndarray, np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]:
        """
        z轴scale

        Args:
            points: 归一化后的点云 (b, n, 3)
            params: normalize_pc_pair返回的参数字典

        Returns:
            scaled_points: z轴scale后的点云
            params: 更新后的参数字典
        """
        # input_data = input_data.detach().cpu().numpy()  # (B, n, 3)
        # gt_data = gt_data.detach().cpu().numpy()  # (B, n, 3)
        # scale = params['scale'].detach().cpu().numpy() # (B, 1, 3)
        # 只对z轴进行scale：z from 0~ to 0~1
        furthest_distances_z = np.amax(
            np.abs(gt_data[..., 2]),
            axis=1,
            keepdims=True,  # B, n, 1
        )  # (b, 1, 1)
        gt_data[..., 2] = gt_data[..., 2] / furthest_distances_z
        input_data[..., 2] = input_data[..., 2] / furthest_distances_z
        if params is not None:
            params["scale"][..., 2] = furthest_distances_z
            return input_data, gt_data, params
        else:
            return input_data, gt_data

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate(
                [idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])]
            )
        return pc[idx[:n]]


def get_datasets(cfg, args, fast_dev_run=False):
    tr_dataset = SMLMDataset(
        cfg=cfg, split="train", fast_dev_run=fast_dev_run, input_dim=3
    )

    eval_split = getattr(args, "eval_split", "val")
    te_dataset = SMLMDataset(
        cfg=cfg, split=eval_split, fast_dev_run=fast_dev_run, input_dim=3
    )
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args, fast_dev_run=False):
    tr_dataset, te_dataset = get_datasets(cfg, args, fast_dev_run)
    train_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        dataset=te_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    loaders = {
        "test_loader": test_loader,
        "train_loader": train_loader,
    }
    return loaders
