import torch
import torch.nn as nn

def physics_smoothness_loss(ddf):
    dy = torch.abs(ddf[:, :, 1:, :] - ddf[:, :, :-1, :])
    dx = torch.abs(ddf[:, :, :, 1:] - ddf[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.contiguous().view(-1), y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1.0 - dice

class MasterPipelineLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_smooth=0.1):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.lambda_dice = lambda_dice
        self.lambda_smooth = lambda_smooth

    def forward(self, grade_logits, target_labels, warped_mask, cc_mask, ddf):
        cls_loss = self.classification_loss(grade_logits, target_labels)
        align_loss = self.dice_loss(warped_mask, cc_mask)
        smooth_loss = physics_smoothness_loss(ddf)
        total = cls_loss + (self.lambda_dice * align_loss) + (self.lambda_smooth * smooth_loss)
        return total, cls_loss, align_loss