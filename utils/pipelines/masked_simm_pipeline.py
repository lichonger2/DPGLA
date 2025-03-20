import os
import numpy as np
import torch
import torch.nn.functional as F

import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss, LovaszSoftmaxLoss, MSELoss
from utils.datasets.visualization import ConfidenceTracker, ConfidenceStatisticsTracker
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
from sklearn.neighbors import BallTree
import open3d as o3d
from scipy.stats import norm
from utils.pseudo.pseudo_filter import DynamicThresholding
from utils.ICP.draw_figure import PseudoLabelStatistics





class SimMaskedAdaptation(pl.core.LightningModule):
    def __init__(self,
                 student_model,
                 teacher_model,
                 momentum_updater,
                 training_dataset,
                 source_validation_dataset,
                 target_validation_dataset,
                 optimizer_name="SGD",
                 source_criterion='CELoss',
                 target_criterion='LovaszSoftmaxLoss',
                 other_criterion=None,
                 source_weight=0.5,
                 target_weight=0.5,
                 filtering=None,
                 lr=1e-3,
                 train_batch_size=12,
                 val_batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None,
                 update_every=1,
                 weighted_sampling=False,
                 target_confidence_th=0.6,
                 selection_perc=0.5,
                 save_mix=False
                 ):

        super().__init__()
        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        self.ignore_label = self.training_dataset.ignore_label

        # ########### LOSSES ##############
        if source_criterion == 'CELoss':
            self.source_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif source_criterion == 'SoftDICELoss':
            self.source_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif source_criterion == 'LovaszSoftmaxLoss':
            self.source_criterion = LovaszSoftmaxLoss(ignore_label=self.training_dataset.ignore_label)
        elif source_criterion == 'FocalLoss':
            self.source_criterion = FocalLoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        # ########### LOSSES ##############
        if target_criterion == 'CELoss':
            self.target_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif target_criterion == 'SoftDICELoss':
            self.target_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif target_criterion == 'LovaszSoftmaxLoss':
            self.target_criterion = LovaszSoftmaxLoss(ignore_label=self.training_dataset.ignore_label)
        elif target_criterion == 'FocalLoss':
            self.target_criterion = FocalLoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        self.other_criterion = other_criterion

        # ############ WEIGHTS ###############
        self.source_weight = source_weight
        self.target_weight = target_weight

        # ############ LABELS ###############
        self.ignore_label = self.training_dataset.ignore_label

        # init
        self.save_hyperparameters(ignore=['teacher_model', 'student_model',
                                          'training_dataset', 'source_validation_dataset',
                                          'target_validation_dataset'])

        # others
        self.validation_phases = ['source_validation', 'target_validation']
        # self.validation_phases = ['pseudo_target']

        self.class2mixed_names = self.training_dataset.class2names
        self.class2mixed_names = np.append(self.class2mixed_names, ["target_label"], axis=0)

        self.voxel_size = self.training_dataset.voxel_size

        # self.knn_search = KNN(k=self.propagation_size, transpose_mode=True)

        if self.training_dataset.weights is not None and self.weighted_sampling:
            tot = self.source_validation_dataset.weights.sum()
            self.sampling_weights = 1 - self.source_validation_dataset.weights/tot

        else:
            self.sampling_weights = None

        self.tracker = ConfidenceTracker(num_classes=19, num_bins=10)
        self.gloabl_tracker = ConfidenceStatisticsTracker(max_iterations=1000)

        self.stats = PseudoLabelStatistics(num_classes=19)

        self.pseudo_filter = DynamicThresholding(num_classes=13, ema_factor=0.1, initial_conf_mean=0.5,
                                                 initial_conf_var=0.05, global_initial_conf_mean=0.85, global_initial_conf_var=0.025)


    @property
    def momentum_pairs(self):
        """Defines base momentum pairs that will be updated using exponential moving average.
        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """
        return [(self.student_model, self.teacher_model)]

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    def random_sample(self, points, sub_num):
        """
        :param points: input points of shape [N, 3]
        :return: np.ndarray of N' points sampled from input points
        """
        """
        随机采样点云数据

        Args:
          points: 输入点云数据，形状为 [N, 3]
          sub_num: 采样点数量

        Returns:
          np.ndarray: 采样后的点云数据索引
        """

        num_points = points.shape[0]

        if sub_num is not None:
            if sub_num <= num_points:
                sampled_idx = np.random.choice(np.arange(num_points), sub_num, replace=False)
            else:
                over_idx = np.random.choice(np.arange(num_points), sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx


    @staticmethod
    def switch_off(labels, switch_classes):
        for s in switch_classes:
            class_idx = labels == s
            labels[class_idx] = -1

        return labels

    def sample_classes(self, origin_classes, num_classes, is_source=False):

        if not is_source:
            if self.weighted_sampling and self.sampling_weights is not None:

                sampling_weights = self.sampling_weights[origin_classes] * (1/self.sampling_weights[origin_classes].sum())

                selected_classes = np.random.choice(origin_classes, num_classes,
                                                    replace=False, p=sampling_weights)

            else:
                selected_classes = np.random.choice(origin_classes, num_classes, replace=False)

        else:
            selected_classes = origin_classes

        return selected_classes


    def height_aware_jitter(self, points, sigma=0.02, clip=0.1, z_threshold_low=0.2, z_threshold_high=0.8):
        """
        Apply height-aware jitter to the points by adding Gaussian noise.

        Args:
            points (np.ndarray): Original points, shape (N, 3).
            sigma (float): Standard deviation of the Gaussian noise.
            clip (float): Maximum amplitude of the Gaussian noise.
            z_threshold_low (float): Lower threshold for normalized z coordinate.
            z_threshold_high (float): Upper threshold for normalized z coordinate.

        Returns:
            np.ndarray: Jittered points.
        """

        if points.size == 0:
            return points

        jittered_points = np.copy(points)

        z_coords = points[:, 2]
        z_min = z_coords.min()
        z_max = z_coords.max()
        if z_max == z_min:
            normalized_z = np.zeros_like(z_coords)
        else:
            normalized_z = (z_coords - z_min) / (z_max - z_min)

        high_z_indices = normalized_z > z_threshold_high
        low_z_indices = normalized_z < z_threshold_low
        mid_z_indices = (normalized_z >= z_threshold_low) & (normalized_z <= z_threshold_high)

        jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)

        jittered_points[high_z_indices, 2] += jitter[high_z_indices, 2]

        jittered_points[low_z_indices, 0] += jitter[low_z_indices, 0]
        jittered_points[low_z_indices, 1] += jitter[low_z_indices, 1]

        jittered_points[mid_z_indices] += jitter[mid_z_indices]

        return jittered_points

    def nonlinear_distance_aware_jitter_with_randomness(self, points, base_sigma=0.005, max_sigma=0.05, clip=0.1):
        """
        Apply nonlinear distance-aware jitter to the points by adding Gaussian noise with added randomness.
        The noise increases nonlinearly with the distance from the origin, and randomness is introduced
        to prevent fixed noise patterns across scans.

        Args:
            points (np.ndarray): Original points, shape (N, 3).
            base_sigma (float): Base noise level for points close to the Lidar.
            max_sigma (float): Maximum noise level for points far from the Lidar.
            clip (float): Maximum amplitude of the Gaussian noise.
            n_parts (int): Number of distance partitions to introduce more randomness.

        Returns:
            np.ndarray: Jittered points.
        """

        if points.size == 0:
            return points

        n_parts = 10

        n_parts = np.random.choice(n_parts)

        distances = np.linalg.norm(points[:, :3], axis=1)

        distances_max = max(distances.max(), 1e-6)  # 避免除零
        normalized_distances = distances / distances_max

        sigma_values = base_sigma + (max_sigma - base_sigma) * np.sqrt(normalized_distances)

        for i in range(1, n_parts + 1):
            part_mask = (normalized_distances >= (i - 1) / n_parts) & (normalized_distances < i / n_parts)
            random_factor = np.random.uniform(0.9, 1.1)
            sigma_values[part_mask] *= random_factor

        jitter = np.clip(sigma_values[:, np.newaxis] * np.random.randn(len(points), 3), -clip, clip)
        jittered_points = points + jitter

        return jittered_points


    def random_global_dropout(self, points, labels, features, drop_rate_min=0.4, drop_rate_max=0.6):
        """
        Apply random dropout to the points, labels, and features with a randomly chosen drop rate.

        Args:
            points (np.ndarray): Input points, shape (N, D).
            labels (np.ndarray): Corresponding labels for the points, shape (N,).
            features (np.ndarray): Corresponding features for the points, shape (N, F).
            drop_rate_min (float): Minimum drop rate.
            drop_rate_max (float): Maximum drop rate.

        Returns:
            tuple: Dropped points, labels, and features.
        """
        num_points = points.shape[0]
        drop_rate = np.random.uniform(drop_rate_min, drop_rate_max)
        keep_indices = np.random.rand(num_points) > drop_rate
        return points[keep_indices], labels[keep_indices], features[keep_indices]


    def laserMix(self, pts1, labels1, features1, pts2, labels2, features2):
        """
        LaserMix that performs point cloud swapping between two sets of data
        based on the pitch angle of points with multiple regions, maintaining
        the behavior of the original laserMix.

        Args:
            pts1 (np.ndarray): First set of point cloud data (target data), shape (N1, D).
            labels1 (np.ndarray): Labels for the first set of point cloud data, shape (N1,).
            features1 (np.ndarray): Features for the first set of point cloud data, shape (N1, F).
            pts2 (np.ndarray): Second set of point cloud data (source data), shape (N2, D).
            labels2 (np.ndarray): Labels for the second set of point cloud data, shape (N2,).
            features2 (np.ndarray): Features for the second set of point cloud data, shape (N2, F).
            pitch_angles (tuple): A tuple containing the lower and upper pitch angle limits for swapping.
            num_areas_options (tuple): Possible numbers of areas to divide the pitch angle range into.

        Returns:
            tuple: Swapped point clouds, labels, and features.
        """
        pitch_angles = (-25, 7)
        num_areas_options = (3, 4, 5, 6)

        pitch_angle_down, pitch_angle_up = pitch_angles
        pitch_angle_down_rad = np.radians(pitch_angle_down)
        pitch_angle_up_rad = np.radians(pitch_angle_up)

        # Compute pitch angles for each point in both point clouds
        rho1 = np.sqrt(pts1[:, 0] ** 2 + pts1[:, 1] ** 2)
        pitch1 = np.arctan2(pts1[:, 2], rho1)

        rho2 = np.sqrt(pts2[:, 0] ** 2 + pts2[:, 1] ** 2)
        pitch2 = np.arctan2(pts2[:, 2], rho2)

        # Randomly choose the number of areas to divide the pitch angle range into
        num_areas = np.random.choice(num_areas_options)
        angle_list = np.linspace(pitch_angle_up_rad, pitch_angle_down_rad, num_areas + 1)

        points_mix1, points_mix2 = [], []
        labels_mix1, labels_mix2 = [], []
        features_mix1, features_mix2 = [], []

        for i in range(num_areas):
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]

            idx1 = np.where((pitch1 > start_angle) & (pitch1 <= end_angle))[0]
            idx2 = np.where((pitch2 > start_angle) & (pitch2 <= end_angle))[0]


            min_len = min(len(idx1), len(idx2))
            if min_len == 0:
                continue  # 如果没有有效数据，跳过

            idx1 = idx1[:min_len]
            idx2 = idx2[:min_len]

            if i % 2 == 0:  # For even areas, pick points from original clouds
                points_mix1.append(pts1[idx1])
                labels_mix1.append(labels1[idx1])
                features_mix1.append(features1[idx1])

                points_mix2.append(pts2[idx2])
                labels_mix2.append(labels2[idx2])
                features_mix2.append(features2[idx2])
            else:  # For odd areas, swap the points
                points_mix1.append(pts2[idx2])
                labels_mix1.append(labels2[idx2])
                features_mix1.append(features2[idx2])

                points_mix2.append(pts1[idx1])
                labels_mix2.append(labels1[idx1])
                features_mix2.append(features1[idx1])

        pts1_out = np.concatenate(points_mix1, axis=0) if points_mix1 else pts1
        labels1_out = np.concatenate(labels_mix1, axis=0) if labels_mix1 else labels1
        features1_out = np.concatenate(features_mix1, axis=0) if features_mix1 else features1

        pts2_out = np.concatenate(points_mix2, axis=0) if points_mix2 else pts2
        labels2_out = np.concatenate(labels_mix2, axis=0) if labels_mix2 else labels2
        features2_out = np.concatenate(features_mix2, axis=0) if features_mix2 else features2

        return pts1_out, labels1_out, features1_out, pts2_out, labels2_out, features2_out

    def soft_dropout(self, points, labels, features, dropout_rate):
        """
        Apply random dropout to the points, labels, and features with a randomly chosen drop rate.

        Args:
            points (np.ndarray): Input points, shape (N, D).
            labels (np.ndarray): Corresponding labels for the points, shape (N,).
            features (np.ndarray): Corresponding features for the points, shape (N, F).
            drop_rate_min (float): Minimum drop rate.
            drop_rate_max (float): Maximum drop rate.

        Returns:
            tuple: Dropped points, labels, and features.
        """
        num_points = points.shape[0]
        # Randomly choose a drop rate between drop_rate_min and drop_rate_max
        drop_rate = 1 - dropout_rate
        keep_indices = np.random.rand(num_points) > drop_rate
        return points[keep_indices], labels[keep_indices], features[keep_indices]



    def group_data_by_bins(self, pts, labels, features, bins):
        bin_dict = {}
        for bin_idx in np.unique(bins):
            bin_dict[bin_idx] = {
                "pts": pts[bins == bin_idx],
                "labels": labels[bins == bin_idx],
                "features": features[bins == bin_idx]
            }
        return bin_dict


    def distribution_adjustment_by_distance(self, origin_pts, origin_labels, origin_features,
                                            dest_pts, dest_labels, dest_features,
                                            tolerance_range=(0.9, 1.1)):
        """
        分布调整函数，将两个点云数据按欧式距离划分为等份，并调整每个距离区间的点云数量，使两者尽可能一致。

        Args:
            origin_pts (np.ndarray): 第一个点云数据，形状为 (N1, 3)。
            origin_labels (np.ndarray): 对应的标签，形状为 (N1, )。
            origin_features (np.ndarray): 对应的特征，形状为 (N1, F)。
            dest_pts (np.ndarray): 第二个点云数据，形状为 (N2, 3)。
            dest_labels (np.ndarray): 对应的标签，形状为 (N2, )。
            dest_features (np.ndarray): 对应的特征，形状为 (N2, F)。
            distance_interval (float): 距离区间的大小（单位：米），默认为 10。
            tolerance_range (tuple): 随机柔性 Dropout 的范围，默认是 (0.95, 1.05)。

        Returns:
            tuple: 调整后的两个点云数据和标签、特征。
        """

        # 计算欧式距离

        distance_interval = 5
        distances1 = np.linalg.norm(origin_pts[:, :3], axis=1)
        distances2 = np.linalg.norm(dest_pts[:, :3], axis=1)

        bins1 = np.floor(distances1 / distance_interval).astype(int)
        bins2 = np.floor(distances2 / distance_interval).astype(int)

        # 将点、标签、特征按照距离区间进行分组


        bin_dict1 = self.group_data_by_bins(origin_pts, origin_labels, origin_features, bins1)
        bin_dict2 = self.group_data_by_bins(dest_pts, dest_labels, dest_features, bins2)

        adjusted_pts1, adjusted_pts2 = [], []
        adjusted_labels1, adjusted_labels2 = [], []
        adjusted_features1, adjusted_features2 = [], []

        # 遍历所有的距离区间
        for bin_key in set(bin_dict1.keys()).union(bin_dict2.keys()):
            pts1_in_bin = bin_dict1.get(bin_key, {}).get("pts", [])
            labels1_in_bin = bin_dict1.get(bin_key, {}).get("labels", [])
            features1_in_bin = bin_dict1.get(bin_key, {}).get("features", [])

            pts2_in_bin = bin_dict2.get(bin_key, {}).get("pts", [])
            labels2_in_bin = bin_dict2.get(bin_key, {}).get("labels", [])
            features2_in_bin = bin_dict2.get(bin_key, {}).get("features", [])

            len1 = len(pts1_in_bin)
            len2 = len(pts2_in_bin)

            min_len = min(len1, len2)
            if min_len == 0:
                continue  # 如果没有有效数据，跳过

            # 柔性 Dropout
            if len1 > len2:
                target_len = int(len2 * np.random.uniform(*tolerance_range))
                dropout_rate = min(1, (target_len / len1))

                dropped_pts1, dropped_labels1, dropped_features1 = self.soft_dropout(pts1_in_bin, labels1_in_bin,
                                                                                       features1_in_bin,
                                                                                       dropout_rate
                                                                                       )


                adjusted_pts1.append(dropped_pts1)
                adjusted_labels1.append(dropped_labels1)
                adjusted_features1.append(dropped_features1)

                adjusted_pts2.append(pts2_in_bin)
                adjusted_labels2.append(labels2_in_bin)
                adjusted_features2.append(features2_in_bin)

            elif len2 > len1:
                target_len = int(len1 * np.random.uniform(*tolerance_range))
                dropout_rate = min(1, (target_len / len2))
                dropped_pts2, dropped_labels2, dropped_features2 = self.soft_dropout(pts2_in_bin, labels2_in_bin,
                                                                                       features2_in_bin,
                                                                                       dropout_rate
                                                                                       )


                adjusted_pts2.append(dropped_pts2)
                adjusted_labels2.append(dropped_labels2)
                adjusted_features2.append(dropped_features2)

                adjusted_pts1.append(pts1_in_bin)
                adjusted_labels1.append(labels1_in_bin)
                adjusted_features1.append(features1_in_bin)
            else:
                adjusted_pts1.append(pts1_in_bin)
                adjusted_labels1.append(labels1_in_bin)
                adjusted_features1.append(features1_in_bin)

                adjusted_pts2.append(pts2_in_bin)
                adjusted_labels2.append(labels2_in_bin)
                adjusted_features2.append(features2_in_bin)

        # 合并调整后的点云、标签、特征
        adjusted_pts1 = np.concatenate(adjusted_pts1, axis=0) if adjusted_pts1 else origin_pts
        adjusted_labels1 = np.concatenate(adjusted_labels1, axis=0) if adjusted_labels1 else origin_labels
        adjusted_features1 = np.concatenate(adjusted_features1, axis=0) if adjusted_features1 else origin_features

        adjusted_pts2 = np.concatenate(adjusted_pts2, axis=0) if adjusted_pts2 else dest_pts
        adjusted_labels2 = np.concatenate(adjusted_labels2, axis=0) if adjusted_labels2 else dest_labels
        adjusted_features2 = np.concatenate(adjusted_features2, axis=0) if adjusted_features2 else dest_features

        return adjusted_pts1, adjusted_labels1, adjusted_features1, adjusted_pts2, adjusted_labels2, adjusted_features2



    def mask(self, origin_pts, origin_labels, origin_features,
             dest_pts, dest_labels, dest_features, use_lasermix=False, is_source=False):
        """
                对点云数据进行掩码处理

                Args:
                  origin_pts: 原始点云数据
                  origin_labels: 原始标签
                  origin_features: 原始特征
                  dest_pts: 目标点云数据
                  dest_labels: 目标标签
                  dest_features: 目标特征
                  is_pseudo: 是否为伪标签

                Returns:
                  tuple: 更新后的目标点云数据、标签、特征和掩码
                """

        origin_pts, origin_labels, origin_features, dest_pts, dest_labels, dest_features=self.distribution_adjustment_by_distance(origin_pts, origin_labels, origin_features,
             dest_pts, dest_labels, dest_features,tolerance_range=(0.9, 1.1))

        if is_source:
            origin_pts = self.nonlinear_distance_aware_jitter_with_randomness(origin_pts)

        else:
            dest_pts = self.nonlinear_distance_aware_jitter_with_randomness(dest_pts)


        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = np.unique(origin_labels)
            origin_present_classes = origin_present_classes[origin_present_classes != -1]

            selected_classes = origin_present_classes

            selected_idx = []
            selected_pts = []
            selected_labels = []
            selected_features = []

            if not self.training_dataset.augment_mask_data:
                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]

                    selected_idx.append(class_idx)
                    selected_pts.append(origin_pts[class_idx])
                    selected_labels.append(origin_labels[class_idx])
                    selected_features.append(origin_features[class_idx])

                if len(selected_pts) > 0:
                    # selected_idx = np.concatenate(selected_idx, axis=0)
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            else:

                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]
                    class_pts = origin_pts[class_idx]
                    class_pts = self.height_aware_jitter(class_pts)
                    num_pts = class_pts.shape[0]
                    sub_num = int(0.99 * num_pts)
                    random_idx = self.random_sample(class_pts, sub_num=sub_num)
                    class_idx = class_idx[random_idx]
                    class_pts = class_pts[random_idx]
                    voxel_mtx, affine_mtx = self.training_dataset.mask_voxelizer.get_transformation_matrix()
                    rigid_transformation = affine_mtx @ voxel_mtx
                    homo_coords = np.hstack((class_pts, np.ones((class_pts.shape[0], 1), dtype=class_pts.dtype)))
                    class_pts = homo_coords @ rigid_transformation.T[:, :3]
                    class_labels = np.ones_like(origin_labels[class_idx]) * sc
                    class_features = origin_features[class_idx]
                    selected_idx.append(class_idx)
                    selected_pts.append(class_pts)
                    selected_labels.append(class_labels)
                    selected_features.append(class_features)

                if len(selected_pts) > 0:#如果有选定的点
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            if len(selected_pts) > 0:
                dest_idx = dest_pts.shape[0]

                if use_lasermix:

                    dest_pts, dest_labels, dest_features, _, _, _ = self.laserMix(
                            dest_pts, dest_labels, dest_features,
                            selected_pts, selected_labels, selected_features
                    )


                else:

                    dest_pts = np.concatenate([dest_pts, selected_pts], axis=0)
                    dest_labels = np.concatenate([dest_labels, selected_labels], axis=0)
                    dest_features = np.concatenate([dest_features, selected_features], axis=0)

                mask = np.ones(dest_pts.shape[0])
                mask[:dest_idx] = 0

            if self.training_dataset.augment_data:
                voxel_mtx, affine_mtx = self.training_dataset.voxelizer.get_transformation_matrix()
                rigid_transformation = affine_mtx @ voxel_mtx
                homo_coords = np.hstack((dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype)))
                dest_pts = homo_coords @ rigid_transformation.T[:, :3]
                duplicated_dest_pts = np.copy(dest_pts)
                duplicated_dest_labels = np.copy(dest_labels)
                duplicated_dest_features = np.copy(dest_features)
                dest_pts = self.height_aware_jitter(dest_pts)
                dest_pts = np.concatenate([dest_pts, duplicated_dest_pts], axis=0)
                dest_labels = np.concatenate([dest_labels, duplicated_dest_labels], axis=0)
                dest_features = np.concatenate([dest_features, duplicated_dest_features], axis=0)
                dest_pts, dest_labels, dest_features = self.random_global_dropout(dest_pts, dest_labels,
                                                                              dest_features)
        return dest_pts, dest_labels, dest_features, mask.astype(np.bool_)

    def mask_data(self, batch, is_oracle=False):
        """
        对批次数据进行掩码处理

        Args:
          batch: 批次数据
          is_oracle: 是否使用真实标签

        Returns:
          dict: 掩码后的新批次数据
        """
        # source
        batch_source_pts = batch['source_coordinates'].cpu().numpy()
        batch_source_labels = batch['source_labels'].cpu().numpy()
        batch_source_features = batch['source_features'].cpu().numpy()

        # target
        batch_target_idx = batch['target_coordinates'][:, 0].cpu().numpy()#目标点云批次索引
        batch_target_pts = batch['target_coordinates'].cpu().numpy()


        batch_target_features = batch['target_features'].cpu().numpy()

        batch_size = int(np.max(batch_target_idx).item() + 1)#获取批次大小

        if is_oracle:
            batch_target_labels = batch['target_labels'].cpu().numpy()#真实标签

        else:
            batch_target_labels = batch['pseudo_labels'].cpu().numpy()#伪标签

        new_batch = {'masked_target_pts': [],
                     'masked_target_labels': [],
                     'masked_target_features': [],
                     'masked_source_pts': [],
                     'masked_source_labels': [],
                     'masked_source_features': []}

        target_order = np.arange(batch_size)

        for b in range(batch_size):
            source_b_idx = batch_source_pts[:, 0] == b
            target_b = target_order[b]#
            target_b_idx = batch_target_idx == target_b

            source_pts = batch_source_pts[source_b_idx, 1:] * self.voxel_size
            source_labels = batch_source_labels[source_b_idx]
            source_features = batch_source_features[source_b_idx]


            target_pts = batch_target_pts[target_b_idx, 1:] * self.voxel_size
            target_labels = batch_target_labels[target_b_idx]
            target_features = batch_target_features[target_b_idx]

            masked_target_pts, masked_target_labels, masked_target_features, masked_target_mask = self.mask(origin_pts=source_pts,
                                                                                                            origin_labels=source_labels,
                                                                                                            origin_features=source_features,
                                                                                                            dest_pts=target_pts,
                                                                                                            dest_labels=target_labels,
                                                                                                            dest_features=target_features,
                                                                                                            use_lasermix=True,
                                                                                                            is_source=True
                                                                                                            )
            # mask后的源数据
            masked_source_pts, masked_source_labels, masked_source_features, masked_source_mask = self.mask(origin_pts=target_pts,
                                                                                                            origin_labels=target_labels,
                                                                                                            origin_features=target_features,
                                                                                                            dest_pts=source_pts,
                                                                                                            dest_labels=source_labels,
                                                                                                            dest_features=source_features,
                                                                                                            use_lasermix=True,
                                                                                                            is_source=False
                                                                                                            )

            if self.save_mix:
                os.makedirs('trial_viz_mix_paper', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/s2t', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/t2s', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/source', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/target', exist_ok=True)
                source_pcd = o3d.geometry.PointCloud()
                valid_source = source_labels != -1
                source_pcd.points = o3d.utility.Vector3dVector(source_pts[valid_source])
                source_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[source_labels[valid_source]+1])

                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(target_pts)
                target_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[target_labels+1])

                s2t_pcd = o3d.geometry.PointCloud()
                s2t_pcd.points = o3d.utility.Vector3dVector(masked_target_pts)
                s2t_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[masked_target_labels+1])

                t2s_pcd = o3d.geometry.PointCloud()
                valid_source = masked_source_labels != -1
                t2s_pcd.points = o3d.utility.Vector3dVector(masked_source_pts[valid_source])
                t2s_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[masked_source_labels[valid_source]+1])
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/source/{self.trainer.global_step}_{b}.ply', source_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/target/{self.trainer.global_step}_{b}.ply', target_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/s2t/{self.trainer.global_step}_{b}.ply', s2t_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/t2s/{self.trainer.global_step}_{b}.ply', t2s_pcd)

                os.makedirs('trial_viz_mix_paper/s2t_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/t2s_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/source_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/target_mask', exist_ok=True)
                source_pcd.paint_uniform_color([1, 0.706, 0])
                target_pcd.paint_uniform_color([0, 0.651, 0.929])

                s2t_pcd = o3d.geometry.PointCloud()
                s2t_pcd.points = o3d.utility.Vector3dVector(masked_target_pts)
                s2t_colors = np.zeros_like(masked_target_pts)
                s2t_colors[masked_target_mask] = [1, 0.706, 0]
                s2t_colors[np.logical_not(masked_target_mask)] = [0, 0.651, 0.929]
                s2t_pcd.colors = o3d.utility.Vector3dVector(s2t_colors)

                t2s_pcd = o3d.geometry.PointCloud()
                valid_source = masked_source_labels != -1
                t2s_pcd.points = o3d.utility.Vector3dVector(masked_source_pts[valid_source])
                t2s_colors = np.zeros_like(masked_source_pts[valid_source])
                masked_source_mask = masked_source_mask[valid_source]
                t2s_colors[masked_source_mask] = [0, 0.651, 0.929]
                t2s_colors[np.logical_not(masked_source_mask)] = [1, 0.706, 0]
                t2s_pcd.colors = o3d.utility.Vector3dVector(t2s_colors)

                o3d.io.write_point_cloud(f'trial_viz_mix_paper/source_mask/{self.trainer.global_step}_{b}.ply', source_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/target_mask/{self.trainer.global_step}_{b}.ply', target_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/s2t_mask/{self.trainer.global_step}_{b}.ply', s2t_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/t2s_mask/{self.trainer.global_step}_{b}.ply', t2s_pcd)
            _, _, _, masked_target_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_target_pts,
                                                                          features=masked_target_features,
                                                                          labels=masked_target_labels,
                                                                          quantization_size=self.training_dataset.voxel_size,
                                                                          return_index=True)

            _, _, _, masked_source_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_source_pts,
                                                                      features=masked_source_features,
                                                                      labels=masked_source_labels,
                                                                      quantization_size=self.training_dataset.voxel_size,
                                                                      return_index=True)
            masked_target_pts = masked_target_pts[masked_target_voxel_idx]
            masked_target_labels = masked_target_labels[masked_target_voxel_idx]
            masked_target_features = masked_target_features[masked_target_voxel_idx]

            masked_source_pts = masked_source_pts[masked_source_voxel_idx]
            masked_source_labels = masked_source_labels[masked_source_voxel_idx]
            masked_source_features = masked_source_features[masked_source_voxel_idx]
            masked_target_pts = np.floor(masked_target_pts/self.training_dataset.voxel_size)
            masked_source_pts = np.floor(masked_source_pts/self.training_dataset.voxel_size)
            batch_index = np.ones([masked_target_pts.shape[0], 1]) * b
            masked_target_pts = np.concatenate([batch_index, masked_target_pts], axis=-1)

            batch_index = np.ones([masked_source_pts.shape[0], 1]) * b
            masked_source_pts = np.concatenate([batch_index, masked_source_pts], axis=-1)

            new_batch['masked_target_pts'].append(masked_target_pts)
            new_batch['masked_target_labels'].append(masked_target_labels)
            new_batch['masked_target_features'].append(masked_target_features)
            new_batch['masked_source_pts'].append(masked_source_pts)
            new_batch['masked_source_labels'].append(masked_source_labels)
            new_batch['masked_source_features'].append(masked_source_features)

        for k, i in new_batch.items():
            if k in ['masked_target_pts', 'masked_target_features', 'masked_source_pts', 'masked_source_features']:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(self.device)
            else:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))

        return new_batch

    def training_step(self, batch, batch_idx):#单个训练步骤
        '''
        :param batch: training batch
        :param batch_idx: batch idx
        :return: None
        '''

        '''
        batch.keys():
            - source_coordinates
            - source_labels
            - source_features
            - source_idx
            - target_coordinates
            - target_labels
            - target_features
            - target_idx
        '''

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        current_epoch = self.current_epoch

        if isinstance(self.target_confidence_th, np.ndarray):
            target_confidence_th = self.target_confidence_th[current_epoch]
            # print("current_target_confidence_th=", target_confidence_th)
        else:
            target_confidence_th = self.target_confidence_th

        target_coord =  batch['target_coordinates']
        target_stensor = ME.SparseTensor(coordinates=batch['target_coordinates'].int(),
                                         features=batch['target_features'])

        target_labels = batch['target_labels'].long().cpu()

        source_stensor = ME.SparseTensor(coordinates=batch['source_coordinates'].int(),
                                         features=batch['source_features'])

        source_labels = batch['source_labels'].long().cpu()

        source_features = source_stensor.F.cpu()
        target_pts = batch['target_coordinates']

        #教师评估模式，使用教师模型前向传播
        self.teacher_model.eval()

        with torch.no_grad():


            target_pseudo = self.teacher_model(target_stensor).F.cpu()
            target_pseudo = F.softmax(target_pseudo, dim=-1)
            target_conf, target_pseudo = target_pseudo.max(dim=-1)
            target_coords = batch['target_coordinates'][:, 1:4].cpu().numpy()
            distances = np.linalg.norm(target_coords, axis=1)
            distances_max = distances.max()
            distances_min = distances.min()
            normalized_distances = (distances - distances_min) / (distances_max - distances_min)
            sigma = 1.0
            weights = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (normalized_distances / sigma) ** 2)
            weights_max = weights.max()
            weights_min = weights.min()
            if weights_max != 0:
                weights = (weights - weights_min) / (weights_max - weights_min)
            distance_weights = torch.tensor(weights, dtype=torch.float32)

            target_conf = target_conf * distance_weights

            #置信度，标签处理
            if self.filtering == 'confidence':

                target_pseudo = self.pseudo_filter.dynamic_confidence_thresholding(
                    target_conf=target_conf,
                    target_pseudo=target_pseudo,
                    conf_threshold=0.5  # 置信度下限 0.3
                )
                self.pseudo_filter.update_stats_conditionally()


            else:
                target_pseudo = F.softmax(target_pseudo, dim=-1)
                target_conf, target_pseudo = target_pseudo.max(dim=-1)


        batch['pseudo_labels'] = target_pseudo
        batch['source_labels'] = source_labels
        masked_batch = self.mask_data(batch, is_oracle=False)


        s2t_stensor = ME.SparseTensor(coordinates=masked_batch["masked_target_pts"].int(),
                                      features=masked_batch["masked_target_features"])

        t2s_stensor = ME.SparseTensor(coordinates=masked_batch["masked_source_pts"].int(),
                                      features=masked_batch["masked_source_features"])


        s2t_labels = masked_batch["masked_target_labels"]
        t2s_labels = masked_batch["masked_source_labels"]
        #学生模型前向传播
        s2t_out = self.student_model(s2t_stensor).F.cpu()
        t2s_out = self.student_model(t2s_stensor).F.cpu()


        s2t_loss = self.target_criterion(s2t_out, s2t_labels.long()) + \
                   self.source_criterion(s2t_out,s2t_labels.long())
        t2s_loss = self.target_criterion(t2s_out, t2s_labels.long()) + \
                   self.source_criterion(t2s_out,t2s_labels.long())

        final_loss = s2t_loss + t2s_loss


        results_dict = {'s2t_loss': s2t_loss.detach(),
                    't2s_loss': t2s_loss.detach(),
                        'final_loss': final_loss.detach()}

        with torch.no_grad():
            self.student_model.eval()
            target_out = self.student_model(target_stensor).F.cpu()
            # 获取每个样本在最后一维上的最大值的索引作为预测类别
            _, target_preds = target_out.max(dim=-1)
            # 计算目标预测与目标标签之间的IoU分数
            target_iou_tmp = jaccard_score(target_preds.numpy(), target_labels.numpy(), average=None,
                                            labels=np.arange(0, self.num_classes),# 计算时考虑的类别范围
                                            zero_division=0.)# 避免除零错误
            # 获取存在于目标标签中的类别和对应的出现次数
            present_labels, class_occurs = np.unique(target_labels.numpy(), return_counts=True)
            # 获取这些存在标签的名称，并为每个名称加上前缀 'student/'
            present_labels = present_labels[present_labels != self.ignore_label]
            # 将学生模型各类别的IoU分数更新到结果字典中
            present_names = self.training_dataset.class2names[present_labels].tolist()
            present_names = ['student/' + p + '_target_iou' for p in present_names]
            results_dict.update(dict(zip(present_names, target_iou_tmp.tolist())))
            results_dict['student/target_iou'] = np.mean(target_iou_tmp[present_labels])

        self.student_model.train()# 将学生模型切换回训练模式
        # 获取有效的伪标签和目标标签的索引
        valid_idx = torch.logical_and(target_pseudo != -1, target_labels != -1) # 计算伪标签与目标标签匹配的数量
        correct = (target_pseudo[valid_idx] == target_labels[valid_idx]).sum()# 计算伪标签的准确率
        pseudo_acc = correct / valid_idx.sum()

        results_dict['teacher/acc'] = pseudo_acc# 更新结果字典中的教师模型伪标签准确率

        results_dict['teacher/confidence'] = target_conf.mean()# 更新结果字典中的教师模型置信度均值

        ann_pts = (target_pseudo != -1).sum()# 计算被标注的点数并更新到结果字典中
        results_dict['teacher/annotated_points'] = ann_pts/target_pseudo.shape[0]

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.train_batch_size,
                add_dataloader_idx=False
            )

        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.
        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """


        if self.trainer.global_step > self.last_step and self.trainer.global_step % self.update_every == 0:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # update tau
            cur_step = self.trainer.global_step
            if self.trainer.accumulate_grad_batches:
                cur_step = cur_step * self.trainer.accumulate_grad_batches
            self.momentum_updater.update_tau(
                cur_step=cur_step,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = self.validation_phases[dataloader_idx]
        # input batch
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.student_model(stensor).F.cpu()

        labels = batch['labels'].long().cpu()
        if phase == 'source_validation':
            loss = self.source_criterion(out, labels)
        else:
            loss = self.target_criterion(out, labels)

        soft_pseudo = F.softmax(out[:, :-1], dim=-1)

        conf, preds = soft_pseudo.max(1)

        iou_tmp = jaccard_score(preds.detach().numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.val_batch_size,
                add_dataloader_idx=False
            )

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            elif self.optimizer_name == 'AdamW':
                optimizer = torch.optim.AdamW(self.student_model.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            elif self.optimizer_name == 'AdamW':
                optimizer = torch.optim.AdamW(self.student_model.parameters(),
                                              lr=self.lr,  # 或者可以尝试 1e-4 到 1e-3
                                              weight_decay=1e-2)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=5,  # 根据你实验的epoch调整
                                                                       eta_min=1e-4)  # 最小学习率
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr / 10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")
            elif self.scheduler_name == 'OneCycleLR':
                steps_per_epoch = int(len(self.training_dataset) / self.train_batch_size)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                                steps_per_epoch=steps_per_epoch,
                                                                epochs=self.trainer.max_epochs)
            else:
                raise NotImplementedError

            return [optimizer], [scheduler]

