from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -----------------------------
# CNN architecture from Part 2
# -----------------------------

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


BASE_DIR = Path(__file__).resolve().parent.parent

RETURN_MODEL_PATH = (
    BASE_DIR / "models" / "return_risk_model.pkl"
)

IMAGE_MODEL_PATH = (
    BASE_DIR / "models" / "product_classifier.pt"
)


# -----------------------------
# Load Return Risk Model
# -----------------------------

return_risk_model = joblib.load(
    RETURN_MODEL_PATH
)


# -----------------------------
# Load Image Classifier
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

image_model = FashionCNN().to(device)

image_model.load_state_dict(
    torch.load(
        IMAGE_MODEL_PATH,
        map_location=device,
    )
)

image_model.eval()


# -----------------------------
# Tool 1: Return Risk
# -----------------------------

def check_return_risk(order_features: dict) -> dict:
    """
    Run the real Part 1 return-risk model.
    """

    order_df = pd.DataFrame(
        [order_features]
    )

    probability = float(
        return_risk_model.predict_proba(
            order_df
        )[0][1]
    )

    prediction = int(
        probability >= 0.44
    )

    return {
        "return_probability": probability,
        "return_probability_percent": round(
            probability * 100,
            2,
        ),
        "prediction": (
            "Likely return"
            if prediction == 1
            else "Likely not return"
        ),
    }


# -----------------------------
# Tool 2: Product Classifier
# -----------------------------

image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])


def classify_product_image(
    image_path: str,
) -> dict:
    """
    Run the real Part 2 CNN on a product image.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(path)

    image_tensor = image_transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predicted_class = (
            probabilities.max(dim=1)
        )

    class_index = predicted_class.item()

    return {
        "predicted_class": CLASS_NAMES[
            class_index
        ],
        "confidence": round(
            confidence.item(),
            4,
        ),
        "confidence_percent": round(
            confidence.item() * 100,
            2,
        ),
        "image_path": str(path),
    }

if __name__ == "__main__":
    print("Testing Part 1 return-risk tool...")

    sample_order = {
        "product_category": "Electronics",
        "price_inr": 16689.0,
        "discount_pct": 6.7,
        "payment_method": "COD",
        "customer_tenure_days": 309,
        "num_previous_orders": 11,
        "num_previous_returns": 3,
        "delivery_distance_km": 166.4,
        "delivery_days": 9,
        "is_weekend_order": 0,
        "rating_given": 2.0,
    }

    result = check_return_risk(
        sample_order
    )

    print(result)

    print("\nTesting Part 2 image classifier...")

    sample_image = (
    BASE_DIR
    / "data"
    / "sample_images"
    / "ankle_boot.png"
)

    if sample_image.exists():
        image_result = classify_product_image(
            str(sample_image)
        )

        print(image_result)

    else:
        print(
            "Sample image not found:"
        )
        print(sample_image)