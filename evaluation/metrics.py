"""
Evaluation metrics for medical image segmentation
"""
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F


def compute_dice_score(predictions, targets, num_classes=9, ignore_background=True):
    """
    Compute Dice Similarity Coefficient for each class
    
    Args:
        predictions: (B, C, H, W) logits or (B, H, W) class indices
        targets: (B, H, W) class indices
        num_classes: number of classes
        ignore_background: whether to ignore background class (class 0)
    
    Returns:
        dice_scores: dict with class names and dice scores
    """
    if predictions.dim() == 4:
        # Convert logits to class predictions
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
    
    dice_scores = {}
    
    for c in range(num_classes):
        if ignore_background and c == 0:
            continue
        
        pred_c = (pred_classes == c).float()
        target_c = (targets == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dice_scores[c] = dice.item()
        else:
            dice_scores[c] = 0.0
    
    return dice_scores


def compute_hausdorff_distance(predictions, targets, num_classes=9, percentile=95, ignore_background=True):
    """
    Compute 95th percentile Hausdorff Distance for each class
    
    Args:
        predictions: (B, C, H, W) logits or (B, H, W) class indices
        targets: (B, H, W) class indices
        num_classes: number of classes
        percentile: percentile for Hausdorff distance (default 95)
        ignore_background: whether to ignore background class
    
    Returns:
        hausdorff_distances: dict with class names and HD95 values
    """
    if predictions.dim() == 4:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
    
    # Convert to numpy
    if isinstance(pred_classes, torch.Tensor):
        pred_classes = pred_classes.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    hausdorff_distances = {}
    
    for c in range(num_classes):
        if ignore_background and c == 0:
            continue
        
        hd_values = []
        
        for b in range(pred_classes.shape[0]):
            pred_mask = (pred_classes[b] == c)
            target_mask = (targets[b] == c)
            
            # Skip if either mask is empty
            if not pred_mask.any() or not target_mask.any():
                continue
            
            # Get boundary points
            pred_boundary = get_boundary_points(pred_mask)
            target_boundary = get_boundary_points(target_mask)
            
            if len(pred_boundary) == 0 or len(target_boundary) == 0:
                continue
            
            # Compute directed Hausdorff distances
            d1 = directed_hausdorff(pred_boundary, target_boundary)[0]
            d2 = directed_hausdorff(target_boundary, pred_boundary)[0]
            hd = max(d1, d2)
            hd_values.append(hd)
        
        if hd_values:
            hd_sorted = sorted(hd_values)
            idx = int(np.ceil(percentile / 100.0 * len(hd_sorted))) - 1
            idx = max(0, min(idx, len(hd_sorted) - 1))
            hausdorff_distances[c] = hd_sorted[idx]
        else:
            hausdorff_distances[c] = float('inf')
    
    return hausdorff_distances


def get_boundary_points(mask):
    """Extract boundary points from binary mask"""
    from scipy.ndimage import binary_erosion
    
    # Erode mask to get interior
    eroded = binary_erosion(mask)
    # Boundary is mask minus interior
    boundary = mask & (~eroded)
    
    # Get coordinates of boundary points
    coords = np.argwhere(boundary)
    return coords


def evaluate_model(model, dataloader, device, num_classes=9):
    """
    Comprehensive evaluation of a model
    
    Returns:
        results: dict with metrics per class and overall
    """
    model.eval()
    
    all_dice_scores = {c: [] for c in range(1, num_classes)}
    all_hd_scores = {c: [] for c in range(1, num_classes)}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Compute metrics per batch
            dice_scores = compute_dice_score(outputs, labels, num_classes=num_classes)
            hd_scores = compute_hausdorff_distance(outputs, labels, num_classes=num_classes)
            
            for c in range(1, num_classes):
                if c in dice_scores:
                    all_dice_scores[c].append(dice_scores[c])
                if c in hd_scores and hd_scores[c] != float('inf'):
                    all_hd_scores[c].append(hd_scores[c])
    
    # Aggregate results
    results = {
        'dice_scores': {},
        'hd95_scores': {},
        'mean_dice': 0.0,
        'mean_hd95': 0.0
    }
    
    dice_values = []
    hd_values = []
    
    for c in range(1, num_classes):
        if all_dice_scores[c]:
            mean_dice = np.mean(all_dice_scores[c])
            std_dice = np.std(all_dice_scores[c])
            results['dice_scores'][c] = {'mean': mean_dice, 'std': std_dice}
            dice_values.append(mean_dice)
        
        if all_hd_scores[c]:
            mean_hd = np.mean(all_hd_scores[c])
            std_hd = np.std(all_hd_scores[c])
            results['hd95_scores'][c] = {'mean': mean_hd, 'std': std_hd}
            hd_values.append(mean_hd)
    
    if dice_values:
        results['mean_dice'] = np.mean(dice_values)
    if hd_values:
        results['mean_hd95'] = np.mean(hd_values)
    
    return results

