import torch
import torch.nn as nn
from . import metrics
from typing import Optional
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# -------------------
class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight

class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)

# -------------------
# --- JaccardLoss ---
# -------------------
class JaccardLoss(nn.Module):
    def __init__(self, class_weights=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "Jaccard"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        return losses


# ----------------
# --- DiceLoss ---
# ----------------
    

def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score




class DiceLoss(nn.Module):
    def __init__(self, class_weights=1.0, smooth=0.05):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "Dice"
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - soft_dice_score(ypr, ygt, self.smooth)  #metrics.iou(ypr, ygt)
        return losses


# ------------------------
# --- CEWithLogitsLoss ---
# ------------------------
class CEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = torch.from_numpy(weight).float() if weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CE"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss


# -----------------
# --- FocalLoss ---
# -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", None]
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.name = "Focal"

    def forward(self, input, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()
        return focal_loss


# ---------------
# --- MCCLoss ---
# ---------------
class MCCLoss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.name = "MCC"

    def forward(self, input, target):
        bs = target.shape[0]

        input = torch.sigmoid(input)

        target = target.view(bs, 1, -1)
        input = input.view(bs, 1, -1)

        tp = torch.sum(torch.mul(input, target)) + self.eps
        tn = torch.sum(torch.mul((1 - input), (1 - target))) + self.eps
        fp = torch.sum(torch.mul(input, (1 - target))) + self.eps
        fn = torch.sum(torch.mul((1 - input), target)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


# ----------------
# --- OHEMLoss ---
# ----------------
class OHEMBCELoss(nn.Module):
    """
    Taken and modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(OHEMBCELoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.name = "OHEM"

    def forward(self, input, target):

        probs = torch.sigmoid(input)[:, 0, :, :].float()
        ygt = target[:, 0, :, :].float()

        # keep hard examples
        kept_flag = torch.zeros_like(probs).bool()
        # foreground pixels with low foreground probability
        kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
        # background pixel with high foreground probability
        kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

        if kept_flag.sum() < self.min_kept:
            # hardest examples have a probability closest to 0.5.
            # The network is very unsure whether they belong to the foreground
            # prob=1 or background prob=0
            hardest_examples = torch.argsort(
                torch.abs(probs - 0.5).contiguous().view(-1)
            )[: self.min_kept]
            kept_flag.contiguous().view(-1)[hardest_examples] = True
        return self.criterion(input[kept_flag, 0], target[kept_flag, 0])


def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)

        nll_loss = -lprobs.gather(dim=dim, index=target.long())
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:

        nll_loss = -lprobs.gather(dim=dim, index=target.long())
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)


    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

# ----------------
# --- SoftCrossEntropyLoss ---
# ----------------
class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=self.dim)
        
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )