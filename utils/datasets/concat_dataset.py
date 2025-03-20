import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from utils.sampling.voxelizer import Voxelizer

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class ConcatDataset(Dataset):
    def __init__(self,
                 source_dataset,
                 target_dataset,
                 augment_mask_data,
                 augment_data,
                 remove_overlap) -> None:
        super().__init__()

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.voxel_size = self.target_dataset.voxel_size

        self.target_len = len(target_dataset)

        self.class2names = self.target_dataset.class2names# 类别名称
        self.colormap = self.target_dataset.color_map

        self.ignore_label = self.target_dataset.ignore_label

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        self.augment_mask_data = augment_mask_data# 是否增强掩码数据
        self.augment_data = augment_data# 是否增强数据
        self.remove_overlap = remove_overlap# 是否移除重叠

        # 数据增强的参数
        self.clip_bounds = None
        self.scale_augmentation_bound = (0.95, 1.05)
        self.rotation_augmentation_bound = ((-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound = None

        self.scale_augmentation_bound_mask = (0.95, 1.05)
        self.rotation_augmentation_bound_mask = (None, None, (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound_mask = None
        # 初始化掩码体素化器
        self.mask_voxelizer = Voxelizer(voxel_size=self.voxel_size,
                                   clip_bound=self.clip_bounds,
                                   use_augmentation=self.augment_mask_data,
                                   scale_augmentation_bound=self.scale_augmentation_bound_mask,
                                   rotation_augmentation_bound=self.rotation_augmentation_bound_mask,
                                   translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound_mask,
                                   ignore_label=vox_ign_label)
        # 初始化体素化器
        self.voxelizer = Voxelizer(voxel_size=self.voxel_size,
                                   clip_bound=self.clip_bounds,
                                   use_augmentation=self.augment_data,
                                   scale_augmentation_bound=self.scale_augmentation_bound,
                                   rotation_augmentation_bound=self.rotation_augmentation_bound,
                                   translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                                   ignore_label=vox_ign_label)

        self.weights = self.source_dataset.weights

    # 体素化方法
    def voxelize(self, data):
        data_pts = data['coordinates']# 获取点云坐标
        data_labels = data['labels']# 获取标签
        data_features = data['features']# 获取特征
        # 使用MinkowskiEngine进行稀疏量化
        _, _, _, voxel_idx = ME.utils.sparse_quantize(coordinates=data_pts,
                                                  features=data_features,
                                                  labels=data_labels,
                                                  quantization_size=self.voxel_size,
                                                  return_index=True)
        ## 根据体素化索引过滤点云数据
        data_pts = data_pts[voxel_idx]/self.voxel_size
        data_labels = data_labels[voxel_idx]# 过滤标签
        data_features = data_features[voxel_idx]# 过滤特征

        if not isinstance(voxel_idx, torch.Tensor):
            voxel_idx = torch.from_numpy(voxel_idx)# 确保体素索引是张量

        return {'coordinates': torch.from_numpy(data_pts).floor(),
                'labels': torch.from_numpy(data_labels),
                'features': torch.from_numpy(data_features),
                'idx': voxel_idx}

    # 合并源数据和目标数据的方法
    def merge(self, source_data, target_data):

        source_data = self.voxelize(source_data)# 体素化源数据
        target_data = self.voxelize(target_data)# 体素化目标数据
        source_data = {f'source_{k}': v for k, v in source_data.items()}
        target_data = {f'target_{k}': v for k, v in target_data.items()}

        data = {**source_data, **target_data}

        return data

    # 获取单个数据项的方法
    def __getitem__(self, idx):
        if idx < len(self.source_dataset):
            source_data = self.source_dataset.get_data(idx)# 从源数据集中获取数据
        else:
            new_idx = np.random.choice(len(self.source_dataset), 1)# 随机选择一个源数据的索引
            source_data = self.source_dataset.get_data(int(new_idx)) # 从源数据集中获取数据

        target_data = self.target_dataset.get_data(idx)# 从目标数据集中获取数据

        return self.merge(source_data, target_data)# 合并源数据和目标数据


    def __len__(self):
        return self.target_len

