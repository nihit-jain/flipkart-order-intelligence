from pathlib import Path

from torchvision import datasets


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "sample_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=False,
)

class_names = {
    0: "tshirt",
    2: "pullover",
    4: "coat",
    6: "shirt",
    9: "ankle_boot",
}

saved = set()

for image, label in dataset:
    if label in class_names and label not in saved:
        filename = OUTPUT_DIR / f"{class_names[label]}.png"

        image.save(filename)

        print(
            f"Saved: {filename} "
            f"(class: {class_names[label]})"
        )

        saved.add(label)

    if len(saved) == len(class_names):
        break

print("\nExport complete.")
print(f"Images saved: {len(saved)}")