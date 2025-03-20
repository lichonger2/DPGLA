import collections
import numpy as np
import MinkowskiEngine as ME
from scipy.linalg import expm, norm

def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=0.05,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):

        self.voxel_size = voxel_size # 体素大小
        self.clip_bound = clip_bound# 裁剪边界
        if ignore_label is not None:
            self.ignore_label = ignore_label
        else:
            self.ignore_label = -100
        # Augmentation
        # 数据增强相关参数
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    # 获取变换矩阵
    def get_transformation_matrix(self):
        """
        生成体素化所需的变换矩阵，包括旋转、缩放和平移矩阵

        Returns:
          voxelization_matrix: 体素化矩阵 (4x4)
          rotation_matrix: 旋转矩阵 (4x4)
        """
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)# 初始化为单位矩阵

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        # 将点云坐标转换为体素坐标
        # 1. 随机旋转
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1# 设置旋转轴
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)# 随机生成旋转角度
                    rot_mats.append(M(axis, theta))# 生成旋转矩阵
                # Use random order
                # 随机顺序
                np.random.shuffle(rot_mats)# 随机打乱旋转矩阵的顺序
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]# 合成最终的旋转矩阵
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat# 将旋转矩阵赋值给变换矩阵
        # 2. Scale and translate to the voxel space.
        # 2. 缩放并平移到体素空间
        scale = 1
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound) # 随机生成缩放因子
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)# 设置缩放矩阵

        # 3. Translate
        # 3. 平移
        if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
            tr = [np.random.uniform(*t) for t in self.translation_augmentation_ratio_bound]# 随机生成平移量
            rotation_matrix[:3, 3] = tr# 将平移量赋值给变换矩阵
        # Get final transformation matrix.
        # 获取最终变换矩阵
        return voxelization_matrix, rotation_matrix

    # 裁剪点云数据
    def clip(self, coords, center=None, trans_aug_ratio=None):
        """
        裁剪点云数据，删除超出边界的点

        Args:
          coords: 点云数据的坐标 (N, 3)
          center: 中心点，默认为 None
          trans_aug_ratio: 平移增强比率，默认为 None

        Returns:
          clip_inds: 裁剪后的索引
        """
        bound_min = np.min(coords, 0).astype(float)# 获取坐标的最小值
        bound_max = np.max(coords, 0).astype(float)# 获取坐标的最大值
        bound_size = bound_max - bound_min# 计算坐标范围
        if center is None:
            center = bound_min + bound_size * 0.5# 设置中心点为范围中心
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)# 根据增强比率计算平移量
            center += trans# 更新中心点
        lim = self.clip_bound# 获取裁剪边界

        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
                             (coords[:, 0] < (lim + center[0])) & \
                             (coords[:, 1] >= (-lim + center[1])) & \
                             (coords[:, 1] < (lim + center[1])) & \
                             (coords[:, 2] >= (-lim + center[2])) & \
                             (coords[:, 2] < (lim + center[2])))
                return clip_inds
        # 裁剪超出限制的点
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
                     (coords[:, 0] < (lim[0][1] + center[0])) & \
                     (coords[:, 1] >= (lim[1][0] + center[1])) & \
                     (coords[:, 1] < (lim[1][1] + center[1])) & \
                     (coords[:, 2] >= (lim[2][0] + center[2])) & \
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None):
        """
         对点云数据进行体素化处理

         Args:
           coords: 点云数据的坐标 (N, 3)
           feats: 点云数据的特征 (N, C)
           labels: 点云数据的标签 (N,)
           center: 中心点，默认为 None

         Returns:
           coords: 量化后的坐标
           feats: 量化后的特征
           labels: 量化后的标签
         """

        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]# 确保输入数据格式正确


        M_v, M_r = self.get_transformation_matrix() # 获取变换矩阵
        rigid_transformation = M_v # 将体素化矩阵赋值给刚性变换矩阵
        # Apply transformations
        if self.use_augmentation:
            # Get rotation and scale
            rigid_transformation = M_r @ rigid_transformation# 应用旋转和缩放矩阵
        # 将齐次坐标应用变换矩阵
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
        coords = homo_coords @ rigid_transformation.T[:, :3]

        # key = self.hash(coords_aug)  # floor happens by astype(np.uint64)
        coords, feats, labels = ME.utils.sparse_quantize(coords,
                                                         feats,
                                                         labels=labels,
                                                         ignore_label=self.ignore_label,
                                                         quantization_size=self.voxel_size)
        return coords, feats, labels