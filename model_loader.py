import joblib
import torch
import torch.nn as nn
from torchvision import datasets, transforms


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

def predict_return_risk(model, order_data):
    probability = model.predict_proba(
        order_data
    )[0][1]

    prediction = int(probability >= 0.44)

    return probability, prediction

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

def predict_image(model, image, device):
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = probabilities.max(dim=1)

    return (
        CLASS_NAMES[predicted_class.item()],
        confidence.item(),
    )

def main() -> None:
    # Load Random Forest return-risk model
    return_risk_model = joblib.load(
        "models/return_risk_model.pkl"
    )

    print("Return-risk model loaded successfully.")

    # Load CNN product classifier
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_model = FashionCNN().to(device)

    image_model.load_state_dict(
        torch.load(
            "models/product_classifier.pt",
            map_location=device,
        )
    )

    image_model.eval()

    print("Image classifier loaded successfully.")
    print("Using device:", device)

    # -----------------------------
    # Return-risk prediction
    # -----------------------------

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

    import pandas as pd

    order_df = pd.DataFrame([sample_order])

    probability, prediction = predict_return_risk(
        return_risk_model,
        order_df,
    )

    print(f"Return probability: {probability:.2%}")
    print(
        "Prediction:",
        "Likely return" if prediction == 1
        else "Likely not return",
    )

    # -----------------------------
    # Image prediction
    # -----------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    image, actual_label = test_dataset[0]

    image = image.unsqueeze(0)

    # Test difficult upper-body classes
    target_classes = [0, 2, 4, 6]

    for target in target_classes:
        for index in range(len(test_dataset)):
            _, label = test_dataset[index]

            if label == target:
                image, actual_label = test_dataset[index]

                image = image.unsqueeze(0)

                predicted_class, confidence = predict_image(
                    image_model,
                    image,
                    device,
                )

                print(
                    f"Actual: {CLASS_NAMES[actual_label]} | "
                    f"Predicted: {predicted_class} | "
                    f"Confidence: {confidence:.2%}"
                )

                break

    predicted_class, confidence = predict_image(
        image_model,
        image,
        device,
    )


if __name__ == "__main__":
    main()

