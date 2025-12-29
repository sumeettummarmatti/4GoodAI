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
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = create_model(
            model_name=self.config['model_name'],
            pretrained=False,
            num_classes=self.config['num_classes'],
            device=self.device
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Updated Class Names for NEU-DET
        self.class_names = [
            'crazing', 'inclusion', 'patches', 
            'pitted', 'rolled-in_scale', 'scratches'
        ]
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def preprocess_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image_rgb.copy()
        augmented = self.transform(image=image_rgb)
        tensor = augmented['image'].unsqueeze(0)
        return tensor, original

    def predict(self, image_path, return_attention=False):
        tensor, original = self.preprocess_image(image_path)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = probs.max(0)
        
        pred_class = predicted.item()
        # Ensure we don't index out of bounds if config changed
        pred_label = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class_{pred_class}"
        
        # Create a dictionary of all probabilities
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        result = {
            'prediction': pred_label,
            'confidence': confidence.item(),
            'probabilities': prob_dict
        }
        
        if return_attention:
            result['attention'] = self.get_grad_cam(tensor, pred_class)
            result['original_image'] = original
        
        return result

    def get_grad_cam(self, tensor, target_class):
        # Change from .blocks[-1] to .layer4[-1] for ResNet architectures
        target_layers = [self.model.backbone.layer4[-1]] 
        
        cam = GradCAM(model=self.model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        return grayscale_cam[0, :]

    def visualize_prediction(self, image_path, save_path=None):
        result = self.predict(image_path, return_attention=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Overlay Grad-CAM
        original_normalized = result['original_image'].astype(np.float32) / 255.0
        cam_overlay = show_cam_on_image(original_normalized, result['attention'], use_rgb=True)
        axes[1].imshow(cam_overlay)
        axes[1].set_title(f'Attention Map: {result["prediction"]}')
        axes[1].axis('off')
        
        # Text results
        axes[2].axis('off')
        prob_text = "\n".join([f"{k}: {v:.2%}" for k, v in result['probabilities'].items()])
        info_text = f"Top Prediction: {result['prediction']}\nConfidence: {result['confidence']:.2%}\n\nAll Probabilities:\n{prob_text}"
        axes[2].text(0, 0.5, info_text, fontsize=12, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        return result


def main():
    parser = argparse.ArgumentParser(description='Defect Detection Inference')
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
    print("DEFECT DETECTION INFERENCE")
    print("=" * 60)
    
    if args.image:
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
        # Note: If you want batch support, you would need to add 
        # the batch_inference method back to your class
        image_dir = Path(args.image_dir)
        for img_path in list(image_dir.glob('*.[jp][pn][g]')):
            detector.visualize_prediction(img_path)
    else:
        print("Error: Please provide --image or --image_dir")

if __name__ == "__main__":
    main()
