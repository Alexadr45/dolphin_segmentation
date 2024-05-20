import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import segmentation_models_pytorch as smp
from Inference_config import InferenceConfig

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((InferenceConfig.shape[0], InferenceConfig.shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=InferenceConfig.normallize_mean, std=InferenceConfig.normallize_std),
    ])
    preprocessed_image = preprocess(image)
    return preprocessed_image.unsqueeze(0)  # Добавляем размерность пакета

model = smp.create_model(
    InferenceConfig.model,
    encoder_name=InferenceConfig.encoder_name,
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

model.load_state_dict(torch.load(InferenceConfig.model_path))
model.eval()

def predict_mask(image_path, model, threshold=InferenceConfig.threshold):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    mask = torch.sigmoid(output).squeeze().numpy()  # Преобразование к массиву NumPy
    mask[mask>=threshold]=1
    mask[mask<threshold]=0
    return mask

def overlay_mask(image_path, mask, alpha, output_path):
    image = Image.open(image_path).convert("RGBA").resize([256, 256])
    mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    red_channel = np.array(mask)
    red_mask = np.stack([red_channel, np.zeros_like(red_channel), np.zeros_like(red_channel), np.ones_like(red_channel) * int(255 * alpha)], axis=-1)
    combined = Image.fromarray(np.uint8(red_mask), 'RGBA')
    overlay = Image.alpha_composite(image, combined)
    overlay.save(output_path.replace('.jpg', '.png'))

def process_images_in_directory(directory_path, model, alpha=alpha, output_directory='output'):
    os.makedirs(output_directory, exist_ok=True)  # Создание выходного каталога, если его нет
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            predicted_mask = predict_mask(image_path, model)
            output_path = os.path.join(output_directory, filename)
            overlay_mask(image_path, predicted_mask, alpha, output_path)

process_images_in_directory(directory_path=InferenceConfig.directory_path, model=model, output_directory=InferenceConfig.output_directory)