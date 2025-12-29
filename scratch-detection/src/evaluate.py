"""
Modified Evaluation script for multi-class scratch/defect detection
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
    """Evaluation class for multi-class defect detection"""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("=" * 60)
        print("DEFECT DETECTION EVALUATION")
        print("=" * 60)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Set dynamic class names based on config
        if self.config['num_classes'] == 6:
            self.class_names = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled-in_scale', 'scratches']
        else:
            self.class_names = ['No-Scratch', 'Scratch']

        # Create model
        self.model = create_model(
            model_name=self.config['model_name'],
            pretrained=False,
            num_classes=self.config['num_classes'],
            device=self.device
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded (Classes: {self.config['num_classes']})")
        
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
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        self.print_classification_report(all_labels, all_preds)
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # ROC and PR curves are typically binary. For multi-class, we skip or 
        # use One-vs-Rest. For now, we save the core metrics.
        self.save_metrics(all_labels, all_preds, all_probs)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE - Results in results/ folder")

    def print_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, 
                                       target_names=self.class_names,
                                       digits=4)
        print("\n" + report)
        
        report_path = Path('results/reports/classification_report.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = Path('results/confusion_matrix/confusion_matrix.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, y_true, y_pred, y_probs):
        from sklearn.metrics import accuracy_score
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }
        
        metrics_path = Path('results/reports/metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    evaluator = Evaluator(model_path='models/best_model.pth')
    evaluator.evaluate()

if __name__ == "__main__":
    main()