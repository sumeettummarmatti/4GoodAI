import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from inference import ScratchDetectorInference
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Initialize detector
try:
    detector = ScratchDetectorInference(model_path='models/best_model.pth')
except Exception as e:
    print(f"Error loading model: {e}. Ensure 'models/best_model.pth' exists.")

def predict_and_visualize(input_img):
    if input_img is None:
        return None, "No image uploaded"

    # Save Gradio's numpy image temporarily for the inference class
    temp_path = "temp_inference.jpg"
    # Convert RGB to BGR for OpenCV compatibility if needed, 
    # but since your class uses cv2.imread, we just save it first.
    Image.fromarray(input_img).save(temp_path)
    
    # 2. Run your existing inference logic
    try:
        result = detector.predict(temp_path, return_attention=True)
        
        # 3. Prepare Grad-CAM overlay
        # result['attention'] is the grayscale heatmap from your get_grad_cam()
        # result['original_image'] is the RGB image from your preprocess_image()
        original_normalized = result['original_image'].astype(np.float32) / 255.0
        cam_overlay = show_cam_on_image(original_normalized, 
                                        result['attention'], 
                                        use_rgb=True)
        
        # 4. Prepare labels
        # result['probabilities'] looks like {'No-Scratch': 0.1, 'Scratch': 0.9}
        return cam_overlay, result['probabilities']
    
    except Exception as e:
        return None, f"Error during processing: {str(e)}"

# 4. Define UI
with gr.Blocks(title="Scratch Detection AI") as demo:
    gr.Markdown("# 🔍 Scratch Detection System")
    gr.Markdown("Upload an image to detect surface defects and see the AI's attention map.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
            submit_btn = gr.Button("Detect Scratch", variant="primary")
        
        with gr.Column():
            output_heatmap = gr.Image(label="Attention Map (Grad-CAM)")
            output_label = gr.Label(label="Predictions")

    submit_btn.click(
        fn=predict_and_visualize,
        inputs=input_image,
        outputs=[output_heatmap, output_label]
    )

if __name__ == "__main__":
    demo.launch()