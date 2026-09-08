import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ocr_dataset import IndianPlateDataset, ID_TO_CHAR, collate_fn
from ocr_model import PlateCRNN


NUM_SAMPLES = 32
BATCH_SIZE = 8
EPOCHS = 400
LEARNING_RATE = 0.0005
RANDOM_SEED = 42


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


def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0
    examples = []

    with torch.no_grad():
        for images, _, _, actual_texts in dataloader:
            images = images.to(device)

            log_probabilities = model(images)
            predicted_texts = greedy_decode(log_probabilities)

            for actual, predicted in zip(actual_texts, predicted_texts):
                if actual == predicted:
                    correct += 1

                total += 1

                if len(examples) < 10:
                    examples.append((actual, predicted))

    return 100 * correct / total, examples


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    full_train_dataset = IndianPlateDataset("train")

    selected_indices = random.sample(
        range(len(full_train_dataset)),
        NUM_SAMPLES
    )

    tiny_dataset = Subset(full_train_dataset, selected_indices)

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    evaluation_loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
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

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for images, labels, label_lengths, _ in tiny_loader:
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

        if epoch == 1 or epoch % 25 == 0:
            accuracy, _ = evaluate(model, evaluation_loader, device)

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Loss: {total_loss / len(tiny_loader):.4f} | "
                f"Exact accuracy on same 32 images: {accuracy:.2f}%"
            )

    final_accuracy, examples = evaluate(model, evaluation_loader, device)

    print("\n--- FINAL SANITY-CHECK RESULTS ---")
    print(f"Exact accuracy on the same 32 images: {final_accuracy:.2f}%")
    print("\nActual plate      Predicted plate")

    for actual, predicted in examples:
        print(f"{actual:<17} {predicted}")


if __name__ == "__main__":
    main()