from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ocr_dataset import create_dataloader, ID_TO_CHAR
from ocr_model import PlateCRNN


BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0005

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_plate_crnn.pt"


def greedy_decode(log_probabilities):
    predicted_ids = log_probabilities.argmax(dim=2).permute(1, 0)

    decoded_texts = []

    for sequence in predicted_ids:
        characters = []
        previous_id = -1

        for character_id in sequence.tolist():
            if character_id != 0 and character_id != previous_id:
                characters.append(ID_TO_CHAR[character_id])

            previous_id = character_id

        decoded_texts.append("".join(characters))

    return decoded_texts


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels, label_lengths, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        log_probabilities = model(images)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probabilities.size(0),
            dtype=torch.long,
            device=device
        )

        loss = criterion(
            log_probabilities,
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_plates = 0
    total_plates = 0

    with torch.no_grad():
        for images, labels, label_lengths, true_texts in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            log_probabilities = model(images)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probabilities.size(0),
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                log_probabilities,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            predicted_texts = greedy_decode(log_probabilities)

            for predicted, actual in zip(predicted_texts, true_texts):
                if predicted == actual:
                    correct_plates += 1

                total_plates += 1

    average_loss = total_loss / len(dataloader)
    exact_accuracy = 100 * correct_plates / total_plates

    return average_loss, exact_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader = create_dataloader(
        split_name="train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = create_dataloader(
        split_name="val",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = PlateCRNN().to(device)

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(validation_loss)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"LR: {current_learning_rate:.6f} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Exact plate accuracy: {validation_accuracy:.2f}%"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "characters": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                },
                MODEL_PATH
            )

            print("Best model saved.")

    print("\nTraining completed.")
    print(f"Best model location: {MODEL_PATH}")


if __name__ == "__main__":
    main()