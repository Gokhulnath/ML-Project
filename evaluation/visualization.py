"""
Visualization utilities for segmentation results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_predictions(images, predictions, targets, num_classes=9, save_path=None):
    """
    Visualize predictions alongside ground truth
    
    Args:
        images: (B, C, H, W) input images
        predictions: (B, C, H, W) logits or (B, H, W) class predictions
        targets: (B, H, W) ground truth labels
        num_classes: number of classes
        save_path: path to save figure
    """
    if predictions.dim() == 4:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
    
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(pred_classes, torch.Tensor):
        pred_classes = pred_classes.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Take first image in batch
    img = images[0].transpose(1, 2, 0) if images[0].shape[0] == 3 else images[0]
    pred = pred_classes[0]
    target = targets[0]
    
    # Create overlay visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(target, cmap='tab20', vmin=0, vmax=num_classes-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='tab20', vmin=0, vmax=num_classes-1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot comparison of metrics across models
    
    Args:
        results_dict: dict with model names as keys and metric results as values
        save_path: path to save figure
    """
    models = list(results_dict.keys())
    dice_scores = [results_dict[m]['mean_dice'] for m in models]
    hd_scores = [results_dict[m]['mean_hd95'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice scores
    ax1.bar(models, dice_scores)
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Mean Dice Score Comparison')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Hausdorff distances
    ax2.bar(models, hd_scores)
    ax2.set_ylabel('HD95 (mm)')
    ax2.set_title('Mean HD95 Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

