import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_ID = {character: index + 1 for index, character in enumerate(CHARACTERS)}
ID_TO_CHAR = {index: character for character, index in CHAR_TO_ID.items()}

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 80


class IndianPlateDataset(Dataset):
    def __init__(self, split_name):
        project_dir = Path(__file__).resolve().parent.parent
        split_dir = project_dir / "data" / "recognition" / "split" / split_name

        self.images_dir = split_dir / "images"
        self.labels_file = split_dir / "labels.csv"

        with open(self.labels_file, "r", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image_path = self.images_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_text = row["label"]
        label_ids = torch.tensor(
            [CHAR_TO_ID[character] for character in label_text],
            dtype=torch.long
        )

        return image, label_ids, label_text


def collate_fn(batch):
    images, label_ids, label_texts = zip(*batch)

    images = torch.stack(images)
    label_lengths = torch.tensor(
        [len(label) for label in label_ids],
        dtype=torch.long
    )
    labels = torch.cat(label_ids)

    return images, labels, label_lengths, label_texts


def create_dataloader(split_name, batch_size=16, shuffle=False):
    dataset = IndianPlateDataset(split_name)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    train_dataset = IndianPlateDataset("train")
    val_dataset = IndianPlateDataset("val")
    test_dataset = IndianPlateDataset("test")

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    image, label_ids, label_text = train_dataset[0]

    print(f"\nOne image tensor shape: {image.shape}")
    print(f"Plate text label: {label_text}")
    print(f"Character ID values: {label_ids.tolist()}")

    train_loader = create_dataloader("train", batch_size=16, shuffle=True)
    images, labels, label_lengths, label_texts = next(iter(train_loader))

    print(f"\nBatch image tensor shape: {images.shape}")
    print(f"Plate labels in first batch: {label_texts[:3]}")
    print(f"Label lengths in first batch: {label_lengths[:3].tolist()}")