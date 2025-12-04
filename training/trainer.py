"""
Training utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from tqdm import tqdm
import os
from .losses import CombinedLoss


class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 150),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Best model tracking
        self.best_val_dice = 0.0
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=config.get('wandb_project', 'medical-segmentation'),
                name=config.get('experiment_name', 'baseline'),
                config=config
            )
        elif config.get('use_wandb', False) and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Continuing without wandb logging.")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Compute Dice score (simplified - full evaluation in evaluation module)
        dice_score = self._compute_dice_score(outputs, labels)
        
        return avg_loss, dice_score
    
    def _compute_dice_score(self, predictions, targets):
        """Compute average Dice score"""
        num_classes = predictions.shape[1]
        probs = torch.softmax(predictions, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for c in range(1, num_classes):  # Skip background
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores.append(dice.item())
        
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('epochs', 150)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_dice = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Dice: {val_dice:.4f}')
            print(f'  LR: {current_lr:.6f}')
            
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'learning_rate': current_lr
                })
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(epoch, is_best=True)
                print(f'  ✓ Saved best model (Dice: {val_dice:.4f})')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f'\nTraining completed! Best validation Dice: {self.best_val_dice:.4f}')
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        return checkpoint['epoch']

