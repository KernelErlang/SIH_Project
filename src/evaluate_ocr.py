from pathlib import Path

import torch

from ocr_dataset import create_dataloader, ID_TO_CHAR
from ocr_model import PlateCRNN


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


def levenshtein_distance(text_a, text_b):
    previous_row = list(range(len(text_b) + 1))

    for index_a, character_a in enumerate(text_a, start=1):
        current_row = [index_a]

        for index_b, character_b in enumerate(text_b, start=1):
            insert_cost = current_row[index_b - 1] + 1
            delete_cost = previous_row[index_b] + 1
            replace_cost = previous_row[index_b - 1] + (
                character_a != character_b
            )

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        previous_row = current_row

    return previous_row[-1]


def evaluate_split(model, split_name, device):
    dataloader = create_dataloader(
        split_name=split_name,
        batch_size=16,
        shuffle=False
    )

    model.eval()

    correct_plates = 0
    total_plates = 0
    total_character_similarity = 0.0
    examples = []

    with torch.no_grad():
        for images, _, _, actual_texts in dataloader:
            images = images.to(device)

            log_probabilities = model(images)
            predicted_texts = greedy_decode(log_probabilities)

            for actual, predicted in zip(actual_texts, predicted_texts):
                total_plates += 1

                if predicted == actual:
                    correct_plates += 1

                maximum_length = max(len(actual), len(predicted), 1)
                distance = levenshtein_distance(actual, predicted)

                similarity = 1 - (distance / maximum_length)
                total_character_similarity += similarity

                if len(examples) < 15:
                    examples.append((actual, predicted, similarity))

    plate_accuracy = 100 * correct_plates / total_plates
    character_accuracy = 100 * total_character_similarity / total_plates

    print(f"\n--- {split_name.upper()} RESULTS ---")
    print(f"Exact plate accuracy: {plate_accuracy:.2f}%")
    print(f"Average character similarity: {character_accuracy:.2f}%")
    print("\nActual plate      Predicted plate      Similarity")

    for actual, predicted, similarity in examples:
        print(f"{actual:<17} {predicted:<20} {similarity * 100:.1f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = PlateCRNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Using device: {device}")
    print(f"Loaded model validation loss: {checkpoint['validation_loss']:.4f}")
    print(
        f"Saved validation exact accuracy: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    evaluate_split(model, "train", device)
    evaluate_split(model, "val", device)
    evaluate_split(model, "test", device)


if __name__ == "__main__":
    main()