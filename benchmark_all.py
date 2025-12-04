"""
Benchmark all algorithms on the Synapse dataset
"""
import argparse
import torch
import time
import os
import json
from torch.utils.data import DataLoader
from data import get_synapse_dataset, get_transforms
from models import UNet, TransUNet, SwinUNet
from training import Trainer
from evaluation import evaluate_model
import pandas as pd


def train_and_evaluate(model_name, data_dir, config, epochs=10):
    """Train a model and return evaluation results"""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get model
    if model_name == 'unet':
        model = UNet(n_channels=3, n_classes=9)
    elif model_name == 'transunet':
        model = TransUNet(img_size=224, in_channels=3, n_classes=9)
    elif model_name == 'swinunet':
        model = SwinUNet(img_size=224, in_channels=3, n_classes=9)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create datasets
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    test_transform = get_transforms(mode='test')
    
    train_dataset = get_synapse_dataset(data_dir, split='train', transform=train_transform, cache=True)
    val_dataset = get_synapse_dataset(data_dir, split='val', transform=val_transform, cache=True)
    test_dataset = get_synapse_dataset(data_dir, split='test', transform=test_transform, cache=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=False)
    
    # Update config
    config['epochs'] = epochs
    config['checkpoint_dir'] = f"checkpoints/{model_name}"
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Load best model
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
    else:
        best_val_dice = 0.0
    
    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"\nEvaluating {model_name} on test set...")
    results = evaluate_model(model, test_loader, device, num_classes=9)
    
    # Measure inference time
    model.eval()
    inference_times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 10:  # Sample 10 batches
                break
            images = images.to(device)
            start = time.time()
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)
    
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    return {
        'model': model_name,
        'parameters': num_params,
        'training_time': training_time,
        'best_val_dice': best_val_dice,
        'test_mean_dice': results['mean_dice'],
        'test_mean_hd95': results['mean_hd95'],
        'inference_time_per_batch': avg_inference_time,
        'dice_scores': results['dice_scores'],
        'hd95_scores': results['hd95_scores']
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark all algorithms')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per model')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    
    # Configuration
    base_config = {
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
        'max_grad_norm': 1.0,
        'use_wandb': False
    }
    
    models_to_train = ['unet', 'transunet', 'swinunet']
    all_results = []
    
    # Train and evaluate each model
    for model_name in models_to_train:
        try:
            config = base_config.copy()
            config['experiment_name'] = f'{model_name}_benchmark'
            
            results = train_and_evaluate(model_name, args.data_dir, config, epochs=args.epochs)
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS - FINAL COMPARISON")
    print(f"{'='*80}\n")
    
    if not all_results:
        print("No models completed training successfully.")
        return
    
    # Summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'].upper(),
            'Parameters': f"{r['parameters']:,}",
            'Training Time (hrs)': f"{r['training_time']/3600:.2f}",
            'Val Dice': f"{r['best_val_dice']:.4f}",
            'Test Dice': f"{r['test_mean_dice']:.4f}",
            'Test HD95 (mm)': f"{r['test_mean_hd95']:.2f}" if r['test_mean_hd95'] != float('inf') else 'N/A',
            'Inference (ms/batch)': f"{r['inference_time_per_batch']*1000:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Find best model
    best_model = max(all_results, key=lambda x: x['test_mean_dice'])
    
    print(f"\n{'='*80}")
    print("WINNER: BEST PERFORMING ALGORITHM")
    print(f"{'='*80}")
    print(f"Model: {best_model['model'].upper()}")
    print(f"Test Dice Score: {best_model['test_mean_dice']:.4f}")
    print(f"Test HD95: {best_model['test_mean_hd95']:.2f} mm" if best_model['test_mean_hd95'] != float('inf') else "Test HD95: N/A")
    print(f"Parameters: {best_model['parameters']:,}")
    print(f"Training Time: {best_model['training_time']/3600:.2f} hours")
    print(f"{'='*80}\n")
    
    # Per-organ comparison
    organ_names = ['aorta', 'gallbladder', 'left_kidney', 'right_kidney', 
                   'liver', 'pancreas', 'spleen', 'stomach']
    
    print("Per-Organ Dice Scores:")
    print("-" * 80)
    print(f"{'Organ':<20}", end="")
    for r in all_results:
        print(f"{r['model'].upper():<15}", end="")
    print()
    print("-" * 80)
    
    for i, organ in enumerate(organ_names, 1):
        print(f"{organ:<20}", end="")
        for r in all_results:
            dice = r['dice_scores'].get(i, {}).get('mean', 0.0)
            print(f"{dice:.4f}          ", end="")
        print()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    df.to_csv('results/benchmark_summary.csv', index=False)
    print(f"\nResults saved to results/benchmark_results.json and results/benchmark_summary.csv")


if __name__ == '__main__':
    main()

