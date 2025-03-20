import os
import torch
import yaml
import numpy as np
import tqdm
from collections import defaultdict

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset
# 获取当前文件的绝对路径
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

# 定义一个SemanticKITTIDataset类，继承自BaseDataset类
class SemanticKITTIDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = '/home/lichonger/AAAAA_DA_semantic_segmentation/ICRA_2025/cosmix-uda/data/SemanticKITTI/sequences',
                 mapping_path: str = '_resources/semantic-kitti.yaml',
                 weights_path: str = None,
                 voxel_size: float = 0.05,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 sub_num: int = 50000,
                 device: str = None,
                 num_classes: int = 19,
                 ignore_label: int = None):
        # 如果权重路径不为空，将其拼接为绝对路径
        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path)
        # 根据版本设置数据集的划分
        if self.version == 'full':
            self.split = {'train': ['00', '01', '02', '03', '04', '05',
                                    '06', '07', '09', '10'],
                          'validation': ['08']}
        elif self.version == 'mini':
            self.split = {'train': ['01'],
                          'validation': ['08']}
        elif self.version == 'sequential':
            self.split = {'train': ['00', '01', '02', '03', '04', '05',
                                    '06', '07', '09', '10'],
                          'validation': ['08']}
        else:
            raise NotImplementedError

        self.name = 'SemanticKITTIDataset'
        # 读取映射文件
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []
        # 创建重映射查找表
        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        # 收集点云和标签文件的路径

        for sequence in self.split[self.phase]:
            num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))

            for f in np.arange(num_frames):
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(f):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)
        # 定义颜色映射
        self.color_map = np.array([(255, 255, 255),  # unlabelled
                                    (25, 25, 255),  # car
                                    (187, 0, 255),  # bicycle
                                    (187, 50, 255),  # motorcycle
                                    (0, 247, 255),  # truck
                                    (50, 162, 168),  # other-vehicle
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # bicyclist
                                    (255, 196, 0),  # motorcyclist
                                    (0, 0, 0),  # road
                                    (148, 148, 148),  # parking
                                    (255, 20, 60),  # sidewalk
                                    (164, 173, 104),  # other-ground
                                    (233, 166, 250),  # building
                                    (255, 214, 251),  # fence
                                    (157, 234, 50),  # vegetation
                                    (107, 98, 56),  # trunk
                                    (78, 72, 44),  # terrain
                                    (83, 93, 130),  # pole
                                    (173, 23, 121)])/255.   # traffic-sign
        # self.voxel_density = []

    def __len__(self):
        # 获取数据集的长度
        return len(self.pcd_path)

    # 获取单个数据项

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i] # 获取第i个点云文件的路径
        label_tmp = self.label_path[i] # 获取第i个标签文件的路径
        # 加载点云数据
        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitti(label_tmp)
        points = pcd[:, :3] # 获取点云坐标

        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        # 将点云数据、颜色数据和标签数据打包成字典
        data = {'points': points, 'colors': colors, 'labels': label}
        # 从字典中获取数据
        points = data['points']
        colors = data['colors']
        labels = data['labels']
        # 初始化采样索引
        sampled_idx = np.arange(points.shape[0])
        # 如果是训练阶段并且需要数据增强，则进行采样和变换
        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]
            # 获取体素化和仿射变换矩阵
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx # 矩阵乘法，得到刚性变换矩阵
            # Apply transformations
            # 应用变换
            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3] # 进行坐标变换
        # 设置忽略标签
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label
        # 使用MinkowskiEngine进行稀疏量化
        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(points,
                                                                               colors,
                                                                               labels=labels,
                                                                               ignore_label=vox_ign_label,
                                                                               quantization_size=self.voxel_size,
                                                                               return_index=True)


        # voxel_density = self.calculate_voxel_density(points)  # 计算体素密度
        # voxel_density_tensor = torch.tensor(list(voxel_density.values()), dtype=torch.float32)  # 将体素密度转换为张量
        # self.voxel_density.append(voxel_density_tensor)
        # 处理缺失点
        missing_pts = self.sub_num - quantized_coords.shape[0]

        # 将numpy数组转换为PyTorch张量
        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)
        # 处理采样索引
        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None
        # 返回处理后的数据字典
        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                # "voxel_density": voxel_density_tensor,
                "idx": torch.tensor(i)
                }

    # def calculate_voxel_density(self, points):
    #     voxel_size = np.array(self.voxel_size) # 获取体素大小
    #     voxel_indices = (points / voxel_size).astype(int) # 计算每个点的体素索引
    #     unique_voxel_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True) # 计算每个体素的点数
    #     voxel_density = defaultdict(int)
    #     for idx, count in zip(unique_voxel_indices, counts):
    #         voxel_density[tuple(idx)] = count # 将点数存入字典
    #
    #     max_density = max(voxel_density.values(), default=1)  # 获取最大密度
    #
    #     for voxel_index in voxel_density:
    #         voxel_density[voxel_index] /= max_density # 归一化密度
    #
    #     return voxel_density # 返回体素密度字典

    # 加载SemanticKITTI标签的方法
    def load_label_kitti(self, label_path: str):
        label = np.fromfile(label_path, dtype=np.uint32) # 从文件中加载标签
        label = label.reshape((-1))  # 重新调整标签形状
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all()) # 确认标签正确
        sem_label = self.remap_lut_val[sem_label] # 进行标签重映射
        return sem_label.astype(np.int32) # 返回处理后的标签

    # 获取数据集权重的方法
    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1) # 初始化权重数组
        for l in tqdm.tqdm(range(len(self.label_path)), desc='Loading weights', leave=True):
            label_tmp = self.label_path[l]
            label = self.load_label_kitti(label_tmp)# 加载标签
            lbl, count = np.unique(label, return_counts=True) # 计算每个标签的数量
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count # 更新权重

        return weights

    # 获取数据的方法
    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]  # 获取第i个点云文件的路径
        label_tmp = self.label_path[i] # 获取第i个标签文件的路径

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitti(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis] # 获取强度信息
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32) # 如果不使用强度信息，则设置为全1
        data = {'points': points, 'colors': colors, 'labels': label}  # 将数据打包成字典

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        # 返回未进行任何变换的数据字典
        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": i}
