import csv
import random
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = PROJECT_DIR / "data" / "recognition" / "images"
SOURCE_LABELS = PROJECT_DIR / "data" / "recognition" / "labels.csv"

OUTPUT_ROOT = PROJECT_DIR / "data" / "recognition" / "split"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42


def read_labels():
    with open(SOURCE_LABELS, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_split(split_name, rows):
    split_folder = OUTPUT_ROOT / split_name
    images_folder = split_folder / "images"
    labels_file = split_folder / "labels.csv"

    images_folder.mkdir(parents=True, exist_ok=True)

    with open(labels_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "label"])
        writer.writeheader()

        for row in rows:
            source_image = SOURCE_IMAGES / row["filename"]
            target_image = images_folder / row["filename"]

            if not source_image.exists():
                raise FileNotFoundError(f"Image not found: {source_image}")

            shutil.copy2(source_image, target_image)
            writer.writerow(row)

    print(f"{split_name}: {len(rows)} images")


def main():
    rows = read_labels()

    if not rows:
        raise ValueError("labels.csv is empty. Generate the plate images first.")

    random.seed(RANDOM_SEED)
    random.shuffle(rows)

    total = len(rows)
    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_rows = rows[:train_end]
    validation_rows = rows[train_end:validation_end]
    test_rows = rows[validation_end:]

    write_split("train", train_rows)
    write_split("val", validation_rows)
    write_split("test", test_rows)

    print("\nDataset split completed.")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()