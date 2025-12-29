"""
Evaluation script for scratch detection
Run: python src/evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve)
from tqdm import tqdm
from pathlib import Path
import json

from data_loader import get_dataloaders
from model import create_model

class Evaluator:
    """Evaluation class for scratch detection"""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("=" * 60)
        print("SCRATCH DETECTION EVALUATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = create_model(
            model_name=self.config['model_name'],
            pretrained=False,
            num_classes=self.config['num_classes'],
            device=self.device
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded (Best Val Acc: {checkpoint['best_val_acc']:.2f}%)")
        
        # Load dataloaders
        _, _, self.test_loader = get_dataloaders(
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            img_size=self.config['img_size']
        )
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION ON TEST SET")
        print("=" * 60)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Collect predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of scratch class
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        self.print_classification_report(all_labels, all_preds)
        self.plot_confusion_matrix(all_labels, all_preds)
        self.plot_roc_curve(all_labels, all_probs)
        self.plot_precision_recall_curve(all_labels, all_probs)
        self.save_metrics(all_labels, all_preds, all_probs)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print("Results saved to results/ directory")
    
    def print_classification_report(self, y_true, y_pred):
        """Print and save classification report"""
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        
        class_names = ['No-Scratch', 'Scratch']
        report = classification_report(y_true, y_pred, 
                                       target_names=class_names,
                                       digits=4)
        print(report)
        
        # Save to file
        report_path = Path('results/reports/classification_report.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("SCRATCH DETECTION - CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        print(f"\n✓ Report saved to {report_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No-Scratch', 'Scratch'],
                   yticklabels=['No-Scratch', 'Scratch'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        save_path = Path('results/confusion_matrix/confusion_matrix.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
        
        # Also plot normalized version
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['No-Scratch', 'Scratch'],
                   yticklabels=['No-Scratch', 'Scratch'],
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path_norm = Path('results/confusion_matrix/confusion_matrix_normalized.png')
        plt.savefig(save_path_norm, dpi=300, bbox_inches='tight')
        print(f"✓ Normalized confusion matrix saved to {save_path_norm}")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_probs):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = Path('results/reports/roc_curve.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_probs):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = Path('results/reports/precision_recall_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to {save_path}")
        plt.close()
    
    def save_metrics(self, y_true, y_pred, y_probs):
        """Save all metrics to JSON"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_probs)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['per_class'] = {
            'no_scratch': {
                'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                'support': int(tn + fp)
            },
            'scratch': {
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                'support': int(tp + fn)
            }
        }
        
        # Save to file
        metrics_path = Path('results/reports/metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics saved to {metrics_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY METRICS")
        print("=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

def main():
    evaluator = Evaluator(model_path='models/best_model.pth')
    evaluator.evaluate()
    
    print("\n[NEXT STEP] Run inference: python src/inference.py --image path/to/image.jpg")

if __name__ == "__main__":
    main()