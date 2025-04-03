import torch
from torchvision import transforms
from models.resnet18.resnet18 import PneumoniaResNet

import gradio as gr
from PIL import Image
import numpy as np
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaResNet.load_from_checkpoint(
    checkpoint_path=Path("checkpoints/model.ckpt"), map_location=device, strict=False
)

print(model)


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

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def analyze_lung_image(input_image):
    try:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        input_tensor = transform(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        diagnosis = pred.item()
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


with gr.Blocks(title="Pneumonia Detection") as demo:
    gr.Markdown("# 🏥 Pneumonia Detection System")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Chest X-Ray", type="pil")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            img_output = gr.Image(label="Processed Image")
            results_output = gr.Textbox(label="Diagnosis Results", interactive=False)

    analyze_btn.click(
        fn=analyze_lung_image, inputs=img_input, outputs=[img_output, results_output]
    )

if __name__ == "__main__":
    demo.launch()
