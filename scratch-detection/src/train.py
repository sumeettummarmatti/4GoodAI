"""
Training script for scratch detection
Run: python src/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
import json
from pathlib import Path

from data_loader import get_dataloaders
from model import create_model, FocalLoss, count_parameters

class Trainer:
    """Training class for scratch detection"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 60)
        print("SCRATCH DETECTION TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size']
        )
        
        # Create model
        self.model = create_model(
            model_name=config['model_name'],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            device=self.device
        )
        
        count_parameters(self.model)
        
        # Loss function
        self.criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [TRAIN]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [VAL]")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss / (pbar.n + 1):.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping
            if epoch - self.best_epoch >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }
        
        save_path = Path('models') / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def save_history(self):
        """Save training history"""
        history_path = Path('results/reports/training_history.json')
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")

def main():
    # Configuration
    config = {
        # Model
        'model_name': 'resnet50',  # CHANGED: Faster convergence
        'pretrained': True,
        'num_classes': 6,  # CHANGED: 6 defect types
        
        # Data
        'batch_size': 32,
        'num_workers': 4,
        'img_size': 224,
        
        # Training
        'epochs': 20,  # CHANGED: Reduced from 50
        'learning_rate': 1e-3,  # CHANGED: Higher LR for faster convergence
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        
        # Loss
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        
        # Early stopping
        'early_stopping_patience': 7  # CHANGED: More aggressive
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train()
    
    print("\n[NEXT STEP] Run evaluation: python src/evaluate.py")

if __name__ == "__main__":
    main()