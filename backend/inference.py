
import io
import torch
from PIL import Image
from torchvision import transforms

from model import model, device, CLASS_NAMES, IMG_SIZE, MEAN, STD


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    probabilities = probabilities[0].detach().cpu().numpy()

    all_probabilities = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "label": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
        "probabilities": all_probabilities
    }
