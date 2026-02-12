from .builder import LOSSES
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import random
import matplotlib.pyplot as plt
unique_names = [
    "smokey", "blaze", "ember", "flare", "ash", "cinder", "sparky",
    "pyro", "charcoal", "scorch", "ignite", "sizzle", "torch", 
    "inferno", "molten", "kindle", "glimmer", "lumin"
]

# Original version from https://github.com/qq456cvb/KeypointNet/blob/master/benchmark_scripts/utils.py#L56
@LOSSES.register_module()
class CorrespondenceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred, pred_offset = pred
        _pred_offset = torch.nn.functional.pad(pred_offset, (1, 0))
        kp_indexs, kp_index_offset = target
        _kp_index_offset = torch.nn.functional.pad(kp_index_offset, (1, 0))
        loss = []
        for j in range(len(kp_index_offset)):
            logits = pred[_pred_offset[j]: _pred_offset[j + 1]]
            kp_index = kp_indexs[_kp_index_offset[j]:_kp_index_offset[j + 1]]  # This can be done with split before the loop
            kp_index = [kp_index[n:n + logits.shape[-1]] for n in range(0, len(kp_index), logits.shape[-1])]
            loss_rot = []
            for rot_kp_index in kp_index:
                loss_rot.append(
                    F.cross_entropy(
                        logits.unsqueeze(0),
                        rot_kp_index.unsqueeze(0).long().cuda(),
                        ignore_index=self.ignore_index,
                    )
                )
            loss.append(torch.min(torch.stack(loss_rot)))
        loss = torch.mean(torch.stack(loss))
        return loss * self.loss_weight


@LOSSES.register_module()
class SaliencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred, pred_offset = pred
        target, target_offset = target
        pred = pred.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, target.to(pred.device))
        return loss * self.loss_weight