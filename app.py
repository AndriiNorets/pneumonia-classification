import torch
from torchvision import transforms
from models.resnet18.resnet18 import PneumoniaResNet
import gradio as gr
from PIL import Image
import numpy as np
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        model = PneumoniaResNet.load_from_checkpoint(
            checkpoint_path=Path("checkpoints/model.ckpt"),
            map_location=device,
            strict=False,
        )
        model = model.to(device).eval()
        print("Model loaded successfully using PyTorch Lightning")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def load_test_images(folder="test_samples"):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                images.append((img, filename))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return images

def analyze_lung_image(input_image):
    try:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        input_tensor = transform(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        diagnosis = "Pneumonia detected" if pred.item() == 1 else "Normal"
        return (
            input_image,
            f"""
            Diagnosis Results:
            - Status: {diagnosis}
            - Confidence: {confidence.item():.2%}
            - Normal Probability: {probs[0][0].item():.2%}
            - Pneumonia Probability: {probs[0][1].item():.2%}
            """,
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        return input_image, "Error processing image. Please try another."

def gallery_select(images, evt: gr.SelectData):
    index = evt.index
    if images and 0 <= index < len(images):
        return images[index][0]
    return None

def clear_selection():
    return None, None, None

test_images = load_test_images()

with gr.Blocks(title="Pneumonia Detection", css=".gallery {min-height: 300px}") as demo:
    gr.Markdown("# 🏥 Pneumonia Detection System")
    
    with gr.Row():
        with gr.Column(scale=1):
            selected_image = gr.Image(label="Selected Image", type="pil", height=300)
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=2):
            results_output = gr.Textbox(label="Diagnosis Results", interactive=False)
            result_image = gr.Image(label="Processed Image", type="pil", height=300)
    
    with gr.Row():
        gallery = gr.Gallery(
            label="Test Images Gallery",
            value=[img[0] for img in test_images],
            columns=4,
            rows=2,
            object_fit="contain",
            height="auto"
        )
    
    gallery.select(
        fn=gallery_select,
        inputs=gr.State(test_images),
        outputs=selected_image
    )
    
    analyze_btn.click(
        fn=analyze_lung_image,
        inputs=selected_image,
        outputs=[result_image, results_output]
    )
    
    clear_btn.click(
        fn=clear_selection,
        outputs=[selected_image, result_image, results_output]
    )

if __name__ == "__main__":
    demo.launch()