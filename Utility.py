from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import torch.nn.functional as F


def replace_batch_to(net, threeD):
    """

    :param net: model
    :param threeD: 3D or not
    :return:
    """
    if threeD:
        for child_name, child in net.named_children():
            if isinstance(child, torch.nn.BatchNorm3d):
                setattr(net, child_name, torch.nn.InstanceNorm3d(child.num_features, track_running_stats=False))
            else:
                replace_batch_to(child, threeD)

    else:
        for child_name, child in net.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                setattr(net, child_name, torch.nn.InstanceNorm2d(child.num_features, track_running_stats=False))
            else:
                replace_batch_to(child, threeD)


# sample for balance dataset
def weight_sampler(data):
    """

    :param data:
    :return:
    """
    targets = data.annotation['Pneumonitis'].to_list()
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.7, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensor s
        inputs = inputs.view(-1)

        targets = targets.view(-1)

        # first compute binary cross-entropy logits
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return focal_loss.mean()
