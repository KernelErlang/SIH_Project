import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ocr_dataset import ID_TO_CHAR, IMAGE_HEIGHT, IMAGE_WIDTH
from ocr_model import PlateCRNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_plate_crnn.pt"


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)


def greedy_decode(log_probabilities):
    predicted_ids = log_probabilities.argmax(dim=2).squeeze(1)

    characters = []
    previous_id = -1

    for character_id in predicted_ids.tolist():
        if character_id != 0 and character_id != previous_id:
            characters.append(ID_TO_CHAR[character_id])

        previous_id = character_id

    return "".join(characters)


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = PlateCRNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint


def predict_plate(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(device)

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        log_probabilities = model(image_tensor)

    predicted_plate = greedy_decode(log_probabilities)

    return predicted_plate, checkpoint, device


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python .\\src\\predict_plate.py path\\to\\plate_image.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    predicted_plate, checkpoint, device = predict_plate(image_path)

    print(f"Using device: {device}")
    print(f"Image: {image_path.name}")
    print(f"Predicted plate: {predicted_plate}")
    print(
        "Saved validation accuracy: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()