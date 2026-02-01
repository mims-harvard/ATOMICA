"""
Custom loss functions for ATOMICA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    Focal loss applies a modulating term to the cross entropy loss in order to focus learning
    on hard misclassified examples. It is defined as:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the true class.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples,
               or a list of weights for each class. If None, no weighting is applied.
        gamma: Focusing parameter >= 0. When gamma=0, this is equivalent to cross-entropy.
               Typically gamma=2.0 works well.
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(8, 5)  # batch_size=8, num_classes=5
        >>> targets = torch.randint(0, 5, (8,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # If alpha is provided as a list/tensor, convert to tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs: Logits from the model, shape (batch_size, num_classes)
            targets: Ground truth class indices, shape (batch_size,)

        Returns:
            loss: Scalar loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probability of the true class for each sample
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Get alpha for each sample based on its true class
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    For multi-label tasks, focal loss is applied independently to each class
    using binary cross entropy with sigmoid activation.

    Args:
        alpha: Weighting factor for positive examples. Can be:
               - float: same weight for all classes
               - list/tensor: per-class weights, shape (num_classes,)
               If None, no weighting is applied.
        gamma: Focusing parameter >= 0. Default: 2.0
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = MultiLabelFocalLoss(gamma=2.0)
        >>> logits = torch.randn(8, 5)  # batch_size=8, num_classes=5
        >>> targets = torch.randint(0, 2, (8, 5)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # If alpha is provided as a list/tensor, convert to tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs: Logits from the model, shape (batch_size, num_classes)
            targets: Ground truth binary labels, shape (batch_size, num_classes)

        Returns:
            loss: Scalar loss value
        """
        # Compute sigmoid probabilities
        p = torch.sigmoid(inputs)

        # Compute binary cross entropy loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute p_t: p if target=1, (1-p) if target=0
        p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Alpha weighting for positive examples
                alpha_t = self.alpha * targets + (1 - targets)
            else:
                # Scalar alpha
                alpha_t = self.alpha * targets + (1 - targets)

            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
