# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from utils import create_ot_matrix




class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss


        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)


        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))


        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class HierachicalOTLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, folder_path: str, is_w_learnable=False, num_classes=1000):
        super().__init__()

        H = create_ot_matrix(folder_path)
        self.num_nodes = H.shape[0]
        # Only keep rows for ImageNet labels (0-999) since other intermediate nodes prediction value is 0 anyway
        H = H[:num_classes, :] 
        H = torch.from_numpy(H).float()
        self.H = Parameter(H, requires_grad=False)
        self.is_w_learnable = is_w_learnable
    
        self.w = Parameter(torch.ones(H.shape[1], dtype=torch.float))

        if self.is_w_learnable:
            print("LEARNABLE OT w!!!")
            self.w.requires_grad = True
        else:
            self.w.requires_grad = False

    def forward(self, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs

        # w.abs(uH - vH) = w.abs((u-v)H)
        diff = labels - outputs
        diff = torch.abs(torch.matmul(diff, self.H))
        ot_loss = (diff @ self.w).mean()

        return ot_loss

