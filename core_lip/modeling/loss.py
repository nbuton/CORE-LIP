import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none" - keeps same shape as BCEWithLogitsLoss


class LDAMLoss(nn.Module):
    def __init__(self, n_pos, n_neg, max_m=0.5, s=30, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.s = s

        # Calculate margins based on 1/n_j^0.25
        m_list = 1.0 / np.sqrt(np.sqrt([n_pos, n_neg]))
        l_cap = max_m / np.max(m_list)
        self.margins = torch.tensor(m_list * l_cap, dtype=torch.float32)

    def forward(self, logits, targets):
        # targets should be 0 or 1
        self.margins = self.margins.to(logits.device)

        # Apply margin: if y=1, subtract margin_pos; if y=0, add margin_neg (to the logit)
        # For binary, we specifically want to penalize the minority class more
        margin_term = self.margins[0] * targets - self.margins[1] * (1 - targets)
        ldam_logits = self.s * (logits - margin_term)

        return F.binary_cross_entropy_with_logits(
            ldam_logits, targets, reduction=self.reduction
        )


class AUCMarginLoss(nn.Module):
    def __init__(self, n_pos, n_neg, margin=1.0, reduction="mean"):
        super().__init__()
        self.p = n_pos / (n_pos + n_neg)
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # We calculate the squared error relative to the margin
        # This is a common surrogate for the AUC ranking objective
        loss_pos = (1 - self.p) * torch.pow(probs - self.margin, 2) * targets
        loss_neg = self.p * torch.pow(probs - (1 - self.margin), 2) * (1 - targets)

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
