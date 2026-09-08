import csv
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "recognition" / "images"
LABELS_FILE = PROJECT_DIR / "data" / "recognition" / "labels.csv"

NUM_IMAGES = 1000
WIDTH = 320
HEIGHT = 80

STATE_CODES = ["WB", "MH", "DL", "KA", "TN", "UP", "BR", "GJ", "RJ", "HR"]

FONT_PATH = r"C:\Windows\Fonts\arialbd.ttf"


def get_font(size):
    if Path(FONT_PATH).exists():
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def make_plate_text():
    state = random.choice(STATE_CODES)
    rto = f"{random.randint(1, 99):02d}"
    series = "".join(random.choices(string.ascii_uppercase, k=random.choice([1, 2])))
    number = f"{random.randint(1, 9999):04d}"

    label = f"{state}{rto}{series}{number}"
    visible_text = f"{state} {rto} {series} {number}"
    return label, visible_text


def draw_centered_text(draw, text, y, font, color):
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    x = (WIDTH - text_width) / 2
    draw.text((x, y), text, font=font, fill=color)


def make_plate_image(visible_text):
    background = random.choice([(255, 255, 255), (250, 250, 220), (245, 245, 245)])
    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)

    draw.rectangle((2, 2, WIDTH - 3, HEIGHT - 3), outline=(0, 0, 0), width=3)

    if random.random() < 0.30:
        parts = visible_text.split()
        draw_centered_text(draw, f"{parts[0]} {parts[1]}", 5, get_font(28), (0, 0, 0))
        draw_centered_text(draw, f"{parts[2]} {parts[3]}", 39, get_font(28), (0, 0, 0))
    else:
        draw_centered_text(draw, visible_text, 22, get_font(32), (0, 0, 0))

    image = image.rotate(
        random.uniform(-4, 4),
        resample=Image.Resampling.BICUBIC,
        fillcolor=background
    )

    if random.random() < 0.40:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])

        for index in range(NUM_IMAGES):
            label, visible_text = make_plate_text()
            image = make_plate_image(visible_text)

            filename = f"plate_{index:05d}.jpg"
            image.save(OUTPUT_DIR / filename, quality=random.randint(75, 95))
            writer.writerow([filename, label])

            if (index + 1) % 100 == 0:
                print(f"Created {index + 1}/{NUM_IMAGES} images")

    print("Dataset generation completed.")
    print(f"Images are in: {OUTPUT_DIR}")
    print(f"Labels are in: {LABELS_FILE}")


if __name__ == "__main__":
    main()