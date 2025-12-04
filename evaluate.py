"""
Evaluation script for trained models
"""
import argparse
import torch
from torch.utils.data import DataLoader
from data import get_synapse_dataset, get_transforms
from models import UNet, TransUNet, SwinUNet
from evaluation import evaluate_model
import json
import os


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
    parser = argparse.ArgumentParser(description='Evaluate medical image segmentation model')
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'transunet', 'swinunet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Synapse dataset directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
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
    
    # Create model
    model = get_model(args.model, n_classes=9)
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    print(f'Best validation Dice: {checkpoint.get("best_val_dice", "N/A"):.4f}')
    
    # Evaluate
    print(f'Evaluating on {args.split} set...')
    results = evaluate_model(model, test_loader, device, num_classes=9)
    
    # Print results
    organ_names = ['background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
                   'liver', 'pancreas', 'spleen', 'stomach']
    
    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)
    print(f'Mean Dice Score: {results["mean_dice"]:.4f}')
    print(f'Mean HD95: {results["mean_hd95"]:.4f} mm')
    print('\nPer-organ results:')
    print('-'*60)
    print(f'{"Organ":<20} {"Dice Score":<15} {"HD95 (mm)":<15}')
    print('-'*60)
    
    for c in range(1, 9):
        organ_name = organ_names[c]
        dice_mean = results['dice_scores'].get(c, {}).get('mean', 0.0)
        dice_std = results['dice_scores'].get(c, {}).get('std', 0.0)
        hd_mean = results['hd95_scores'].get(c, {}).get('mean', float('inf'))
        hd_std = results['hd95_scores'].get(c, {}).get('std', 0.0)
        
        dice_str = f'{dice_mean:.4f} ± {dice_std:.4f}'
        if hd_mean != float('inf'):
            hd_str = f'{hd_mean:.2f} ± {hd_std:.2f}'
        else:
            hd_str = 'N/A'
        
        print(f'{organ_name:<20} {dice_str:<15} {hd_str:<15}')
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f'{args.model}_{args.split}_results.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f'\nResults saved to {results_file}')


if __name__ == '__main__':
    import numpy as np
    main()

