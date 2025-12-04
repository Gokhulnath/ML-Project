"""
Main training script for medical image segmentation models
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data import get_synapse_dataset, get_transforms
from models import UNet, TransUNet, SwinUNet
from training import Trainer


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
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'transunet', 'swinunet'],
                        help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Synapse dataset directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with command line arguments
    config.update({
        'model': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'experiment_name': f'{args.model}_baseline'
    })
    
    # Create datasets
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    train_dataset = get_synapse_dataset(
        data_dir=args.data_dir,
        split='train',
        transform=train_transform,
        cache=True
    )
    
    val_dataset = get_synapse_dataset(
        data_dir=args.data_dir,
        split='val',
        transform=val_transform,
        cache=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = get_model(args.model, n_classes=9)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model}')
    print(f'Number of parameters: {num_params:,}')
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

