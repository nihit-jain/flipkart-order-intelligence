from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RETURN_MODEL_PATH = (
    BASE_DIR / "models" / "return_risk_model.pkl"
)

RESNET_FEATURE_MODEL_PATH = (
    BASE_DIR / "models" / "resnet18_feature_extractor.pt"
)

RESNET_CLASSIFIER_PATH = (
    BASE_DIR / "models" / "resnet_feature_classifier.pkl"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Class names
# --------------------------------------------------

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


# --------------------------------------------------
# Tool 1: Return Risk
# --------------------------------------------------

return_risk_model = joblib.load(
    RETURN_MODEL_PATH
)


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


# --------------------------------------------------
# ResNet18 Feature Extractor
# --------------------------------------------------

weights = ResNet18_Weights.DEFAULT

resnet18 = models.resnet18(
    weights=None
)

resnet = nn.Sequential(
    *list(resnet18.children())[:-1]
)

resnet.load_state_dict(
    torch.load(
        RESNET_FEATURE_MODEL_PATH,
        map_location=device,
    )
)

resnet = resnet.to(device)
resnet.eval()


# --------------------------------------------------
# ResNet-compatible image transform
# --------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])


# --------------------------------------------------
# Logistic Regression classifier
# --------------------------------------------------

resnet_classifier = joblib.load(
    RESNET_CLASSIFIER_PATH
)


# --------------------------------------------------
# Tool 2: Product Image Classification
# --------------------------------------------------

def classify_product_image(
    image_path: str,
) -> dict:
    """
    Run the final Part 2 ResNet18 feature-extraction
    classifier on a product image.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        path
    ).convert("L")

    image_tensor = image_transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
      features = resnet(
        image_tensor
    )

    features = features.view(
        features.size(0),
        -1,
    )

    features = features.cpu().numpy()

    probabilities = resnet_classifier.predict_proba(
        features
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    return {
        "predicted_class": CLASS_NAMES[
            predicted_class
        ],
        "confidence": round(
            confidence,
            4,
        ),
        "confidence_percent": round(
            confidence * 100,
            2,
        ),
        "image_path": str(path),
    }


# --------------------------------------------------
# Local tool test
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "Testing Part 1 return-risk tool..."
    )

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

    print(
        "\nTesting Part 2 ResNet image classifier..."
    )

    sample_image = (
        BASE_DIR
        / "data"
        / "sample_images"
        / "ankle_boot.png"
    )

    if sample_image.exists():

        image_result = (
            classify_product_image(
                str(sample_image)
            )
        )

        print(image_result)

    else:

        print(
            "Sample image not found:"
        )

        print(sample_image)