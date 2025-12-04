"""
Check training progress and show current status
"""
import os
import json
import glob

def check_progress():
    print("="*70)
    print("TRAINING PROGRESS CHECK")
    print("="*70)
    
    models = ['unet', 'transunet', 'swinunet']
    
    for model in models:
        checkpoint_dir = f'checkpoints/{model}'
        print(f"\n{model.upper()}:")
        print("-" * 70)
        
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            if 'best_model.pth' in files:
                checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location='cpu')
                epoch = checkpoint.get('epoch', 'N/A')
                val_dice = checkpoint.get('best_val_dice', 'N/A')
                print(f"  Status: COMPLETE")
                print(f"  Best Epoch: {epoch}")
                print(f"  Best Val Dice: {val_dice:.4f}" if isinstance(val_dice, float) else f"  Best Val Dice: {val_dice}")
            else:
                checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
                if checkpoints:
                    latest = max(checkpoints, key=os.path.getctime)
                    checkpoint = torch.load(latest, map_location='cpu')
                    epoch = checkpoint.get('epoch', 'N/A')
                    print(f"  Status: IN PROGRESS")
                    print(f"  Current Epoch: {epoch}")
                else:
                    print(f"  Status: STARTING...")
        else:
            print(f"  Status: NOT STARTED")
    
    # Check for results
    if os.path.exists('results/final_benchmark_results.json'):
        print(f"\n{'='*70}")
        print("FINAL RESULTS AVAILABLE!")
        print(f"{'='*70}")
        with open('results/final_benchmark_results.json', 'r') as f:
            results = json.load(f)
        
        print("\nSummary:")
        for r in results:
            print(f"\n{r['model']}:")
            print(f"  Test Dice: {r['test_mean_dice']:.4f}")
            print(f"  Test HD95: {r['test_mean_hd95']:.2f} mm" if r['test_mean_hd95'] != float('inf') else "  Test HD95: N/A")
        
        best = max(results, key=lambda x: x['test_mean_dice'])
        print(f"\n🏆 BEST MODEL: {best['model']} (Dice: {best['test_mean_dice']:.4f})")

if __name__ == '__main__':
    import torch
    check_progress()

