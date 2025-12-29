"""
Inference script for scratch detection with Grad-CAM visualization
Run: python src/inference.py --image path/to/image.jpg
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_model

class ScratchDetectorInference:
    """Inference class with visualization"""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        print(f"✓ Model loaded successfully!")
        
        # Transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class names
        self.class_names = ['No-Scratch', 'Scratch']
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Read image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original for visualization
        original = image_rgb.copy()
        
        # Apply transforms
        augmented = self.transform(image=image_rgb)
        tensor = augmented['image'].unsqueeze(0)
        
        return tensor, original
    
    def predict(self, image_path, return_attention=False):
        """Make prediction on single image"""
        # Preprocess
        tensor, original = self.preprocess_image(image_path)
        tensor = tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        # Get class
        pred_class = predicted.item()
        pred_label = self.class_names[pred_class]
        confidence = confidence.item()
        
        # Get probabilities for both classes
        no_scratch_prob = probs[0, 0].item()
        scratch_prob = probs[0, 1].item()
        
        result = {
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': {
                'No-Scratch': no_scratch_prob,
                'Scratch': scratch_prob
            }
        }
        
        if return_attention:
            result['attention'] = self.get_grad_cam(tensor, pred_class)
            result['original_image'] = original
        
        return result
    
    def get_grad_cam(self, tensor, target_class):
        """Generate Grad-CAM heatmap"""
        # Define target layer (last conv layer of backbone)
        target_layers = [self.model.backbone.blocks[-1]]
        
        # Create Grad-CAM
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with Grad-CAM"""
        # Get prediction with attention
        result = self.predict(image_path, return_attention=True)
        
        # Prepare visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM overlay
        original_normalized = result['original_image'].astype(np.float32) / 255.0
        cam_overlay = show_cam_on_image(original_normalized, 
                                        result['attention'], 
                                        use_rgb=True)
        axes[1].imshow(cam_overlay)
        axes[1].set_title('Attention Map (Grad-CAM)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction info
        axes[2].axis('off')
        info_text = f"""
PREDICTION RESULTS
{'=' * 30}

Predicted Class: {result['prediction']}
Confidence: {result['confidence']:.2%}

Class Probabilities:
  • No-Scratch: {result['probabilities']['No-Scratch']:.2%}
  • Scratch: {result['probabilities']['Scratch']:.2%}

{'=' * 30}
Model: {self.config['model_name']}
        """
        axes[2].text(0.1, 0.5, info_text, fontsize=11, 
                    verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def batch_inference(self, image_dir, output_dir='results/sample_predictions'):
        """Run inference on multiple images"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_paths = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.bmp'))
        
        print(f"\nRunning inference on {len(image_paths)} images...")
        
        results = []
        for img_path in image_paths:
            print(f"\nProcessing: {img_path.name}")
            
            # Predict and visualize
            save_path = output_dir / f"prediction_{img_path.stem}.png"
            result = self.visualize_prediction(img_path, save_path)
            
            # Print result
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            
            results.append({
                'image': img_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        
        print(f"\n✓ All predictions saved to {output_dir}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Scratch Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/sample_predictions',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create inference object
    detector = ScratchDetectorInference(model_path=args.model)
    
    print("=" * 60)
    print("SCRATCH DETECTION INFERENCE")
    print("=" * 60)
    
    if args.image:
        # Single image inference
        print(f"\nProcessing image: {args.image}")
        save_path = Path(args.output) / f"prediction_{Path(args.image).stem}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = detector.visualize_prediction(args.image, save_path)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.2%}")
    
    elif args.image_dir:
        # Batch inference
        results = detector.batch_inference(args.image_dir, args.output)
    
    else:
        print("\nError: Please provide --image or --image_dir")
        print("Examples:")
        print("  python src/inference.py --image data/test/image.jpg")
        print("  python src/inference.py --image_dir data/test/")

if __name__ == "__main__":
    main()