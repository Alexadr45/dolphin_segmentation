import torch
import torch.nn.functional as F

class IoUBCELoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target, smooth=1):
        pred = torch.sigmoid(pred)
        iou = iou_loss(pred, target, smooth)
        bce = F.binary_cross_entropy(pred.squeeze(), target.squeeze(), reduction='mean')
        iou_bce = self.weight * bce + iou * (1 - self.weight)

        return iou_bce
