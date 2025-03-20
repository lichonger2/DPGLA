import torch

class DynamicThresholding:
    def __init__(self, num_classes, ema_factor=0.1,
                 initial_conf_mean=0.65, initial_conf_var=0.05,
                 global_initial_conf_mean=0.85, global_initial_conf_var=0.025):
        self.num_classes = num_classes
        self.ema_factor = ema_factor
        # 初始化每个类别的置信度均值和方差
        self.class_conf_mean = {c: initial_conf_mean for c in range(num_classes)}
        self.class_conf_var = {c: initial_conf_var for c in range(num_classes)}
        # 初始化全局置信度均值和方差
        self.global_conf_mean = global_initial_conf_mean
        self.global_conf_var = global_initial_conf_var
        self.iteration = 0
        self.iteration_global = 0
        self.global_ema_factor = ema_factor / 10

    def update_conf_stats(self, class_id, new_conf_mean, new_conf_var):

        self.class_conf_mean[class_id] = (1 - self.ema_factor) * self.class_conf_mean[class_id] + \
                                         self.ema_factor * new_conf_mean
        self.class_conf_var[class_id] = (1 - self.ema_factor) * self.class_conf_var[class_id] + \
                                        self.ema_factor * new_conf_var

    def update_global_conf_stats(self, new_global_mean, new_global_var):

        self.global_conf_mean = (1 - self.global_ema_factor) * self.global_conf_mean + \
                                self.global_ema_factor * new_global_mean
        self.global_conf_var = (1 - self.global_ema_factor) * self.global_conf_var + \
                               self.global_ema_factor * new_global_var

    def dynamic_confidence_thresholding(self, target_conf, target_pseudo, conf_threshold=0.3):

        filtered_target_pseudo = -torch.ones_like(target_pseudo)

        # Step 1: 动态计算全局置信度均值和方差
        valid_target_conf = target_conf[~torch.isnan(target_conf) & ~torch.isinf(target_conf)]
        if valid_target_conf.numel() > 0:
            global_mean = valid_target_conf.mean().item()
            global_var = valid_target_conf.var().item()
            self.update_global_conf_stats(global_mean, global_var)

        # Step 2: 动态设定高置信度阈值
        high_conf_threshold = self.global_conf_mean + self.global_conf_var

        high_conf_idx = target_conf > high_conf_threshold
        filtered_target_pseudo[high_conf_idx] = target_pseudo[high_conf_idx]

        valid_idx = (target_conf > conf_threshold) & (target_conf <= high_conf_threshold)

        for class_id in range(self.num_classes):
            class_idx = (target_pseudo == class_id) & valid_idx

            if class_idx.sum() == 0:
                continue

            class_conf_values = target_conf[class_idx]

            class_conf_values = class_conf_values[~torch.isnan(class_conf_values)]
            class_conf_values = class_conf_values[~torch.isinf(class_conf_values)]

            if class_conf_values.numel() == 0:
                continue

            class_conf_mean = class_conf_values.mean().item()

            if class_conf_values.numel() > 1:
                class_conf_var = class_conf_values.var().item()
            else:
                class_conf_var = 0.0

            self.update_conf_stats(class_id, class_conf_mean, class_conf_var)

            dynamic_threshold = self.class_conf_mean[class_id] - self.class_conf_var[class_id]

            class_valid_idx = class_conf_values > dynamic_threshold
            valid_class_idx = torch.nonzero(class_idx)[class_valid_idx]

            filtered_target_pseudo[valid_class_idx] = target_pseudo[valid_class_idx]

        return filtered_target_pseudo.long()
    def update_stats_conditionally(self):
        """
        控制置信度均值和方差的更新频率。
        前500次迭代每次更新一次，之后每500次迭代更新一次。
        """
        self.iteration += 1
        if self.iteration <= 500:
            self.ema_factor = 1 / (self.iteration + 1)
            self.global_ema_factor = 1 / (self.iteration + 1)
        elif self.iteration % 500 == 0:
            self.ema_factor = 0.1
            self.global_ema_factor = 0.01


