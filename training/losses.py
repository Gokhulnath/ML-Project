"""
Loss functions for medical image segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        """
        num_classes = predictions.shape[1]
        dice_loss = 0.0
        
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        for c in range(num_classes):
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice
        
        return dice_loss / num_classes


class CombinedLoss(nn.Module):
    """Combined Dice Loss and Cross-Entropy Loss"""
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce

