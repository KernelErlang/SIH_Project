import torch
from PIL import Image
from torchvision import transforms
from OCR import PlateCRNN

CRNN_MODEL_PATH = r"CRNN MODEL\best_plate_crnn.pt"
IMAGE_PATH = r"C:\Users\HP\OneDrive\Documents\traffic_platform\REAL DATA FOR CRNN\1.jpeg"

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_ID = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 320

device = torch.device("cpu")

crnn_model = PlateCRNN()
checkpoint = torch.load(CRNN_MODEL_PATH, map_location=device)
print("Validation loss:", checkpoint['validation_loss'])
print("Validation accuracy:", checkpoint['validation_accuracy'])
crnn_model.load_state_dict(checkpoint['model_state_dict'])
crnn_model.eval()

# ---- EXACT same transform as ocr_dataset.py's IndianPlateDataset ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def ctc_greedy_decode(log_probs):
    preds = log_probs.argmax(2).squeeze(1).tolist()
    decoded = []
    prev = None
    for p in preds:
        if p != prev and p != BLANK_ID:
            decoded.append(IDX_TO_CHAR.get(p, "?"))
        prev = p
    return "".join(decoded)


image = Image.open(IMAGE_PATH).convert("RGB")
plate_tensor = transform(image).unsqueeze(0)   # (1, 1, 80, 320)

with torch.no_grad():
    log_probs = crnn_model(plate_tensor)

print("CRNN read:", ctc_greedy_decode(log_probs))