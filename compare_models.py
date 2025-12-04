"""
Script to compare all trained models
"""
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import evaluate_model
from evaluation.visualization import plot_metrics_comparison
import torch
from torch.utils.data import DataLoader
from data import get_synapse_dataset, get_transforms
from models import UNet, TransUNet, SwinUNet


def get_model(model_name, n_classes=9):
    """Get model by name"""
    if model_name.lower() == 'unet':
        return UNet(n_channels=3, n_classes=n_classes)
    elif model_name.lower() == 'transunet':
        return TransUNet(img_size=224, in_channels=3, n_classes=n_classes)
    elif model_name.lower() == 'swinunet':
        return SwinUNet(img_size=224, in_channels=3, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Compare all trained models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Synapse dataset directory')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints (format: model_name:path)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    test_transform = get_transforms(mode='test')
    test_dataset = get_synapse_dataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=test_transform,
        cache=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate all models
    all_results = {}
    organ_names = ['background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
                   'liver', 'pancreas', 'spleen', 'stomach']
    
    for checkpoint_spec in args.checkpoints:
        model_name, checkpoint_path = checkpoint_spec.split(':')
        
        print(f'\nEvaluating {model_name}...')
        
        # Create and load model
        model = get_model(model_name, n_classes=9)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Evaluate
        results = evaluate_model(model, test_loader, device, num_classes=9)
        all_results[model_name] = results
    
    # Create comparison table
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Per-organ comparison
    comparison_data = []
    for model_name, results in all_results.items():
        for c in range(1, 9):
            organ_name = organ_names[c]
            dice_mean = results['dice_scores'].get(c, {}).get('mean', 0.0)
            dice_std = results['dice_scores'].get(c, {}).get('std', 0.0)
            hd_mean = results['hd95_scores'].get(c, {}).get('mean', float('inf'))
            hd_std = results['hd95_scores'].get(c, {}).get('std', 0.0)
            
            comparison_data.append({
                'Model': model_name,
                'Organ': organ_name,
                'Dice_Mean': dice_mean,
                'Dice_Std': dice_std,
                'HD95_Mean': hd_mean if hd_mean != float('inf') else None,
                'HD95_Std': hd_std if hd_mean != float('inf') else None
            })
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(args.output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nComparison table saved to {csv_path}')
    
    # Summary statistics
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Mean_Dice': results['mean_dice'],
            'Mean_HD95': results['mean_hd95']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Summary statistics saved to {summary_path}')
    
    # Create visualization
    plot_path = os.path.join(args.output_dir, 'model_comparison.png')
    plot_metrics_comparison(all_results, save_path=plot_path)
    print(f'Comparison plot saved to {plot_path}')
    
    # Print summary
    print('\n' + '='*60)
    print('Summary Statistics')
    print('='*60)
    print(summary_df.to_string(index=False))
    print('='*60)


if __name__ == '__main__':
    import numpy as np
    main()

