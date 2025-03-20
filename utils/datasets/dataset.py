import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from utils.sampling.voxelizer import Voxelizer

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# 定义一个基础数据集类，继承自PyTorch的Dataset类
class BaseDataset(Dataset):
    def __init__(self,
                 version: str,
                 phase: str,
                 dataset_path: str,
                 voxel_size: float = 0.05,
                 sub_num: int = 50000,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 device: str = None,
                 weights_path: str = None):

        self.CACHE = {}
        self.version = version
        self.phase = phase
        self.dataset_path = dataset_path
        self.voxel_size = voxel_size  # # 体素大小（以米为单位）
        self.sub_num = sub_num # 采样点数
        self.use_intensity = use_intensity # 是否使用强度信息
        self.augment_data = augment_data and self.phase == 'train' # 数据增强（仅在训练阶段使用）
        self.num_classes = num_classes # 类别数量

        self.ignore_label = ignore_label # 忽略的标签
        # 如果忽略标签为None，设置体素化忽略标签为-100
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        # for input augs
        # self.clip_bounds = ((-100, 100), (-100, 100), (-100, 100))
        # 输入数据增强的参数
        # self.clip_bounds = ((-100, 100), (-100, 100), (-100, 100))
        self.clip_bounds = None
        self.scale_augmentation_bound = (0.95, 1.05) # 缩放增强范围
        # 旋转增强范围
        self.rotation_augmentation_bound = ((-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20))
        # 平移增强范围
        self.translation_augmentation_ratio_bound = None
        # 初始化体素化器
        self.voxelizer = Voxelizer(voxel_size=self.voxel_size,
                                   clip_bound=self.clip_bounds,
                                   use_augmentation=self.augment_data,
                                   scale_augmentation_bound=self.scale_augmentation_bound,
                                   rotation_augmentation_bound=self.rotation_augmentation_bound,
                                   translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                                   ignore_label=vox_ign_label)

        self.device = device
        # 数据集划分（训练和验证）
        self.split = {'train': [],
                      'validation': []}

        self.maps = None
        self.color_map = None
        # 如果提供了权重路径，加载权重
        self.weights_path = weights_path
        if self.weights_path is not None:
            self.weights = np.load(self.weights_path)
        else:
            self.weights = None

    # 定义数据集长度方法（需要子类实现）
    def __len__(self):
        raise NotImplementedError

    # 定义获取数据方法（需要子类实现）
    def __getitem__(self, i: int):
        raise NotImplementedError

    def random_sample(self, points: np.ndarray, center: np.array = None) -> np.array:
        """
        :param points: input points of shape [N, 3]
        :param center: center to sample around, default is None, not used for now
        :return: np.ndarray of N' points sampled from input points
        """
        """
        :param points: 输入点云，形状为[N, 3]
        :param center: 采样中心（目前未使用，默认为None）
        :return: 从输入点云中采样的点云数据
        """

        num_points = points.shape[0]# 获取点的数量


        if self.sub_num is not None:
            if self.sub_num <= num_points:
                # 如果采样点数小于等于总点数，随机选择sub_num个点
                sampled_idx = np.random.choice(np.arange(num_points), self.sub_num, replace=False)
            else:
                # 如果采样点数大于总点数，随机选择超出部分的点
                over_idx = np.random.choice(np.arange(num_points), self.sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx

