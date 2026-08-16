import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

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

device = torch.device("cpu")

print("=" * 60)
print("PART 2 - RESNET18 TRANSFER LEARNING")
print("=" * 60)
print("Device:", device)


# ---------------------------------------------------------
# 1. Fashion-MNIST -> ResNet-compatible images
# ---------------------------------------------------------

weights = ResNet18_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

# CPU-friendly subset while preserving all classes.
# The full dataset is already available and can be expanded later.
train_indices = list(range(12000))
test_indices = list(range(3000))

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print("Training samples:", len(train_subset))
print("Test samples:", len(test_subset))


# ---------------------------------------------------------
# 2. Load pretrained ResNet18
# ---------------------------------------------------------

print("\nLoading pretrained ResNet18...")

resnet = models.resnet18(weights=weights)

# Freeze pretrained backbone.
for parameter in resnet.parameters():
    parameter.requires_grad = False

# Remove the original ImageNet classifier.
feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

print("Pretrained ResNet18 loaded.")
print("Backbone frozen.")
print("Feature dimension: 512")


# ---------------------------------------------------------
# 3. Feature extraction
# ---------------------------------------------------------

def extract_features(model, loader):
    features = []
    labels = []

    with torch.no_grad():
        for batch_index, (images, batch_labels) in enumerate(loader):
            images = images.to(device)

            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)

            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.numpy())

            if (batch_index + 1) % 50 == 0:
                print(
                    f"Extracted batches: "
                    f"{batch_index + 1}/{len(loader)}"
                )

    return (
        np.concatenate(features),
        np.concatenate(labels),
    )


print("\nExtracting training features...")
X_train, y_train = extract_features(
    feature_extractor,
    train_loader,
)

print("\nExtracting test features...")
X_test, y_test = extract_features(
    feature_extractor,
    test_loader,
)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ---------------------------------------------------------
# 4. Train classifier on extracted ResNet features
# ---------------------------------------------------------

print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=300,
    solver="lbfgs",
    random_state=SEED,
)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("\n" + "=" * 60)
print("FINAL RESNET FEATURE-EXTRACTION EVALUATION")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
    )
)


# ---------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------

np.save(
    os.path.join(
        MODEL_DIR,
        "resnet_train_features.npy",
    ),
    X_train,
)

np.save(
    os.path.join(
        MODEL_DIR,
        "resnet_train_labels.npy",
    ),
    y_train,
)

np.save(
    os.path.join(
        MODEL_DIR,
        "resnet_test_features.npy",
    ),
    X_test,
)

np.save(
    os.path.join(
        MODEL_DIR,
        "resnet_test_labels.npy",
    ),
    y_test,
)

np.save(
    os.path.join(
        MODEL_DIR,
        "resnet_confusion_matrix.npy",
    ),
    cm,
)

joblib.dump(
    classifier,
    os.path.join(
        MODEL_DIR,
        "resnet_feature_classifier.pkl",
    ),
)

torch.save(
    feature_extractor.state_dict(),
    os.path.join(
        MODEL_DIR,
        "resnet18_feature_extractor.pt",
    ),
)

print("\nSaved:")
print("models/resnet_train_features.npy")
print("models/resnet_test_features.npy")
print("models/resnet_feature_classifier.pkl")
print("models/resnet18_feature_extractor.pt")
print("models/resnet_confusion_matrix.npy")

print("\n" + "=" * 60)
print("PART 2 RESNET PIPELINE COMPLETE")
print("=" * 60)