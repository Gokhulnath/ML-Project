"""
Simplified benchmark that trains models sequentially with progress output
"""
import os
# Disable torch compilation to avoid transformers compatibility issue
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import torch
torch._dynamo.config.suppress_errors = True
import time
import os
import json
from torch.utils.data import DataLoader
from data import get_synapse_dataset, get_transforms
from models import UNet, TransUNet, SwinUNet
from training import Trainer
from evaluation import evaluate_model
import pandas as pd


def main():
    data_dir = './dataset'
    epochs = 5  # Reduced for faster results
    batch_size = 2
    
    base_config = {
        'batch_size': batch_size,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
        'max_grad_norm': 1.0,
        'use_wandb': False
    }
    
    models_config = {
        'unet': (UNet(n_channels=3, n_classes=9), 'U-Net'),
        'transunet': (TransUNet(img_size=224, in_channels=3, n_classes=9), 'TransUNet'),
        'swinunet': (SwinUNet(img_size=224, in_channels=3, n_classes=9), 'Swin-UNet')
    }
    
    all_results = []
    
    # Create datasets once
    print("Loading datasets...")
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    test_transform = get_transforms(mode='test')
    
    train_dataset = get_synapse_dataset(data_dir, split='train', transform=train_transform, cache=True)
    val_dataset = get_synapse_dataset(data_dir, split='val', transform=val_transform, cache=True)
    test_dataset = get_synapse_dataset(data_dir, split='test', transform=test_transform, cache=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"Training: {len(train_dataset)} slices, Validation: {len(val_dataset)} slices, Test: {len(test_dataset)} slices\n")
    
    # Train each model
    for model_key, (model, model_name) in models_config.items():
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name}")
        print(f"{'='*70}")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        
        config = base_config.copy()
        config['epochs'] = epochs
        config['checkpoint_dir'] = f"checkpoints/{model_key}"
        config['experiment_name'] = f'{model_key}_benchmark'
        
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
        
        # Evaluate
        print(f"\nEvaluating {model_name} on test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        results = evaluate_model(model, test_loader, device, num_classes=9)
        
        # Measure inference
        inference_times = []
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                if i >= 5:
                    break
                images = images.to(device)
                start = time.time()
                _ = model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append((time.time() - start) / images.shape[0])
        
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        
        result = {
            'model': model_name,
            'parameters': num_params,
            'training_time': training_time,
            'best_val_dice': best_val_dice,
            'test_mean_dice': results['mean_dice'],
            'test_mean_hd95': results['mean_hd95'],
            'inference_time_per_image': avg_inference,
            'dice_scores': {k: v.get('mean', 0) for k, v in results['dice_scores'].items()},
            'hd95_scores': {k: v.get('mean', float('inf')) for k, v in results['hd95_scores'].items()}
        }
        all_results.append(result)
        
        print(f"\n{model_name} Results:")
        print(f"  Validation Dice: {best_val_dice:.4f}")
        print(f"  Test Dice: {results['mean_dice']:.4f}")
        print(f"  Test HD95: {results['mean_hd95']:.2f} mm" if results['mean_hd95'] != float('inf') else "  Test HD95: N/A")
    
    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*70}\n")
    
    # Create summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'],
            'Parameters': f"{r['parameters']:,}",
            'Training Time (min)': f"{r['training_time']/60:.1f}",
            'Val Dice': f"{r['best_val_dice']:.4f}",
            'Test Dice': f"{r['test_mean_dice']:.4f}",
            'Test HD95 (mm)': f"{r['test_mean_hd95']:.2f}" if r['test_mean_hd95'] != float('inf') else 'N/A',
            'Inference (ms/img)': f"{r['inference_time_per_image']*1000:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Find winner
    best = max(all_results, key=lambda x: x['test_mean_dice'])
    
    print(f"\n{'='*70}")
    print("🏆 WINNER: BEST ALGORITHM")
    print(f"{'='*70}")
    print(f"Algorithm: {best['model']}")
    print(f"Test Dice Score: {best['test_mean_dice']:.4f}")
    if best['test_mean_hd95'] != float('inf'):
        print(f"Test HD95: {best['test_mean_hd95']:.2f} mm")
    print(f"Parameters: {best['parameters']:,}")
    print(f"Training Time: {best['training_time']/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    # Per-organ breakdown
    organ_names = {1: 'aorta', 2: 'gallbladder', 3: 'left_kidney', 4: 'right_kidney',
                   5: 'liver', 6: 'pancreas', 7: 'spleen', 8: 'stomach'}
    
    print("Per-Organ Dice Scores:")
    print("-" * 70)
    print(f"{'Organ':<20}", end="")
    for r in all_results:
        print(f"{r['model']:<15}", end="")
    print()
    print("-" * 70)
    
    for i, organ in organ_names.items():
        print(f"{organ:<20}", end="")
        for r in all_results:
            dice = r['dice_scores'].get(i, 0.0)
            print(f"{dice:.4f}          ", end="")
        print()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/final_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    df.to_csv('results/final_benchmark_summary.csv', index=False)
    print(f"\nResults saved to results/final_benchmark_results.json")


if __name__ == '__main__':
    main()

