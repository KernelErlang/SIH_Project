import csv
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "recognition" / "images"
LABELS_FILE = PROJECT_DIR / "data" / "recognition" / "labels.csv"

NUM_IMAGES = 5000
WIDTH = 320
HEIGHT = 80

STATE_CODES = [
    "WB", "MH", "DL", "KA", "TN", "UP", "BR", "GJ",
    "RJ", "HR", "AP", "TS", "KL", "OD", "PB", "MP"
]

FONT_PATHS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\bahnschrift.ttf",
]


def available_fonts():
    fonts = [Path(font) for font in FONT_PATHS if Path(font).exists()]

    if not fonts:
        raise FileNotFoundError(
            "No configured font was found in C:\\Windows\\Fonts."
        )

    return fonts


def create_plate_number():
    state = random.choice(STATE_CODES)
    rto_code = f"{random.randint(1, 99):02d}"
    series_length = random.choice([1, 2])
    series = "".join(
        random.choices(string.ascii_uppercase, k=series_length)
    )
    number = f"{random.randint(1, 9999):04d}"

    label = f"{state}{rto_code}{series}{number}"
    visible_text = f"{state} {rto_code} {series} {number}"

    return label, visible_text


def font_that_fits(draw, text):
    font_path = random.choice(available_fonts())

    for font_size in range(36, 15, -1):
        font = ImageFont.truetype(font_path, font_size)
        box = draw.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]

        if text_width <= WIDTH - 24:
            return font

    return ImageFont.truetype(font_path, 16)


def create_one_line_plate(visible_text):
    background = random.choice([
        (255, 255, 255),
        (252, 252, 230),
        (245, 245, 245)
    ])

    text_color = random.choice([
        (0, 0, 0),
        (15, 15, 15),
        (30, 30, 30)
    ])

    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (2, 2, WIDTH - 3, HEIGHT - 3),
        outline=(0, 0, 0),
        width=3
    )

    font = font_that_fits(draw, visible_text)
    box = draw.textbbox((0, 0), visible_text, font=font)

    text_width = box[2] - box[0]
    text_height = box[3] - box[1]

    x = (WIDTH - text_width) / 2
    y = ((HEIGHT - text_height) / 2) - box[1] - 1

    draw.text((x, y), visible_text, fill=text_color, font=font)

    image = image.rotate(
        random.uniform(-3, 3),
        resample=Image.Resampling.BICUBIC,
        fillcolor=background
    )

    if random.random() < 0.35:
        image = image.filter(
            ImageFilter.GaussianBlur(random.uniform(0.1, 0.7))
        )

    if random.random() < 0.30:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.80, 1.20)
        )

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        for index in range(NUM_IMAGES):
            label, visible_text = create_plate_number()
            image = create_one_line_plate(visible_text)

            filename = f"plate_{index:05d}.jpg"
            image.save(
                OUTPUT_DIR / filename,
                quality=random.randint(80, 95)
            )

            writer.writerow([filename, label])

            if (index + 1) % 500 == 0:
                print(f"Created {index + 1}/{NUM_IMAGES} images")

    print("\nOne-line dataset generation completed.")
    print(f"Images: {OUTPUT_DIR}")
    print(f"Labels: {LABELS_FILE}")


if __name__ == "__main__":
    main()