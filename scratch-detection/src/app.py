import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_model

class DefectApp:
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
        
        # NEU-DET Class Names
        self.class_names = [
            'Crazing', 'Inclusion', 'Patches', 
            'Pitted', 'Rolled-in Scale', 'Scratches'
        ]
        
        # Inference Transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def predict(self, input_img):
        if input_img is None:
            return None, None
        
        # 1. Prepare Image: Resize to 224x224 to avoid broadcasting errors
        # This ensures original_resized and heatmap both have shape (224, 224, 3)
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        original_resized = cv2.resize(input_rgb, (224, 224))
        
        # 2. Preprocess for Model
        augmented = self.transform(image=input_rgb)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 3. Model Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidences = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
            
        # 4. Grad-CAM Visualization
        # Use .layer4[-1] for ResNet architectures (fixes 'blocks' error)
        target_layers = [self.model.backbone.layer4[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        # Generate grayscale heatmap
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0, :]
        
        # Overlay heatmap on resized original image
        img_float = original_resized.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        return cam_image, confidences

# Initialize App Logic
app_logic = DefectApp()

# Define Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Surface Defect Detection System")
    gr.Markdown("Upload a metal surface image to identify defects and visualize the AI's focus area.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Surface Image")
            btn = gr.Button("Detect Defect", variant="primary")
        
        with gr.Column():
            output_cam = gr.Image(label="Attention Map (Grad-CAM)")
            output_labels = gr.Label(label="Predictions", num_top_classes=3)

    btn.click(
        fn=app_logic.predict,
        inputs=input_image,
        outputs=[output_cam, output_labels]
    )

if __name__ == "__main__":
    demo.launch(share=False)