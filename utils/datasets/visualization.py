import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class ConfidenceTracker:
    def __init__(self, num_classes=19, num_bins=10):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.bins = torch.linspace(0, 1, num_bins + 1)
        self.confidence_data = {i: [] for i in range(num_classes)}

    def update(self, confidences, labels):
        """
        Update the tracker with new batch of confidences and labels
        """
        for i in range(self.num_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                self.confidence_data[i].extend(confidences[class_mask].tolist())

    def compute_distribution(self):
        """
        Compute the confidence distribution for each class
        """
        distributions = {}
        for label, confidences in self.confidence_data.items():
            if confidences:
                # Get the histogram count for the current class
                hist, _ = np.histogram(confidences, bins=self.bins.numpy())
                distributions[label] = hist / len(confidences)  # Normalize to get percentages
        return distributions

    def plot_distributions(self, distributions):
        """
        Plot confidence distributions for all labels
        """
        save_dir = '/home/lichonger/AAAAA_DA_semantic_segmentation/ICRA_2025/cosmix-uda/figure'
        os.makedirs(save_dir, exist_ok=True)
        for label, hist in distributions.items():
            plt.figure()
            plt.bar(self.bins[:-1].numpy(), hist, width=0.1, edgecolor='black')
            plt.title(f"Confidence Distribution for Label {label}")
            plt.xlabel("Confidence")
            plt.ylabel("Proportion")
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.savefig(os.path.join(save_dir, f"confidence_distribution_label_{label}.png"))
            plt.close()  # 关闭图形窗口以释放内存


class ConfidenceStatisticsTracker:
    def __init__(self, max_iterations=None):
        """
        初始化统计器
        :param max_iterations: 控制存储的最大迭代数据量（最近 n 个迭代的数据）
        """
        self.global_confidences = []  # 用于存储所有迭代的target_conf
        self.max_iterations = max_iterations  # 限制最多存储的迭代数量（None 表示无限制）

    def update(self, confidences):
        """
        更新置信度数据
        :param confidences: 当前 batch 的置信度 (Tensor)
        """
        self.global_confidences.extend(confidences.tolist())  # 确保追加全局数据
        if self.max_iterations is not None and len(self.global_confidences) > self.max_iterations:
            # 控制全局存储的数据量
            self.global_confidences = self.global_confidences[-self.max_iterations:]
    def compute_statistics(self):
        """
        计算最近 n 个迭代内所有 target_conf 的均值和方差
        """
        if len(self.global_confidences) > 0:
            mean = np.mean(self.global_confidences)
            variance = np.var(self.global_confidences)
            return mean, variance
        return None, None

    def reset(self):
        """
        重置置信度存储
        """
        self.global_confidences = []
