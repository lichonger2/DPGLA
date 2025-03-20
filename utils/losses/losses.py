import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CELoss(nn.Module):
    def __init__(self, ignore_label: int = None, weight: np.ndarray = None):
        '''
        :param ignore_label: label to ignore
        :param weight: possible weights for weighted CE Loss
        '''
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(weight).float()
            print(f'----->Using weighted CE Loss weights: {weight}')

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)
        self.ignored_label = ignore_label

    def forward(self, preds: torch.Tensor, gt: torch.Tensor):

        loss = self.loss(preds, gt)
        return loss


class DICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True):
        super(DICELoss, self).__init__()

        # 如果指定了忽略标签，将其转换为tensor

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize# 是否使用平方和
        self.use_tmask = use_tmask# 是否使用掩码

    def forward(self, output, target):
        input_device = output.device# 获取输入的设备类型
        # temporal solution to avoid nan
        # 暂时解决nan问题，将数据转移到CPU上
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            # 过滤掉需要忽略的标签
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]
        # 将目标标签转换为one-hot编码
        target = F.one_hot(target, num_classes=output.shape[1])

        # 对输出进行softmax处理
        output = F.softmax(output, dim=-1)
        # 计算交集
        intersection = (output * target).sum(dim=0)
        if self.powerize:
            # 如果使用平方和，计算并加上一个小常数以避免除零
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            # 计算掩码，仅计算有目标的类别
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()
        # 计算IoU（交并比），对每个类别求和并归一化
        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)  # 返回DICE损失，并转回原设备


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps# 最大值设为1-eps
    min_val = eps / (t_vector.shape[-1] - 1)# 最小值为eps的均分

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val# 将0位置的值设为最小值
    t_soft[t_vector == 1] = max_val # 将1位置的值设为最大值

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val
    # 针对KITTI数据集，特殊处理部分标签
    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val/2
        t_soft[searched_idx, 6] = max_val/2

    return t_soft


class SoftDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range# 是否使用负范围
        self.eps = eps# epsilon值，用于平滑
        self.is_kitti = is_kitti# 是否是KITTI数据集

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            # 如果使用负范围，DICE损失为负的IoU平均值
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            # 返回DICE损失和每个类别的损失
            return dice_loss.to(input_device), dice_class
        else:
            # 只返回DICE损失
            return dice_loss.to(input_device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

    def forward(self, output, target):
        input_device = output.device
        # 转移到 CPU 进行计算以避免 NaN
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        # 转换为 one-hot 编码
        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        target_onehot = target_onehot.float()

        # 计算 softmax 概率
        prob = F.softmax(output, dim=-1)

        # 计算 focal loss
        focal_weight = self.alpha * (1 - prob) ** self.gamma
        loss = -focal_weight * target_onehot * torch.log(prob + 1e-12)
        loss = loss.sum(dim=1).mean()

        return loss.to(input_device)


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_label=None):
        super(LovaszSoftmaxLoss, self).__init__()
        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

    def forward(self, output, target):
        input_device = output.device
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        # Apply softmax to get probabilities
        output = F.softmax(output, dim=-1)

        # One-hot encoding for the target
        target_onehot = F.one_hot(target, num_classes=output.shape[1]).float()

        # Compute Lovasz-Softmax loss
        loss = self.lovasz_softmax_flat(output, target_onehot)

        return loss.to(input_device)

    def lovasz_softmax_flat(self, prob, labels):
        """
        Compute the Lovasz-Softmax loss
        """
        if prob.numel() == 0:
            return prob * 0.

        # Flatten the input tensors
        prob_flat = prob.view(-1, prob.size(-1))
        labels_flat = labels.view(-1, labels.size(-1))

        # Intersection over union loss
        iou_loss = self.lovasz_softmax(prob_flat, labels_flat)

        return iou_loss

    @staticmethod
    def lovasz_softmax(probs, labels):
        """
        Lovasz-Softmax loss function
        """
        C = probs.size(1)
        losses = []
        for c in range(C):
            fg = labels[:, c]
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(LovaszSoftmaxLoss.lovasz_grad(fg_sorted) @ errors_sorted)
        return torch.mean(torch.stack(losses))

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if gt_sorted.numel() > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        定义 MSE 损失类
        :param reduction: 损失的聚合方式，可选 'mean'（默认），'sum' 或 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        计算 MSE 损失
        :param preds: 模型的预测值 (logits or features)
        :param targets: 增强后的目标值 (logits or features)
        :return: 计算后的 MSE 损失
        """
        loss = F.mse_loss(preds, targets, reduction=self.reduction)
        return loss