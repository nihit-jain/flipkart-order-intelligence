import tempfile
from pathlib import Path
from agent.tools import classify_product_image
from agent.api import router as agent_router
from fastapi.middleware.cors import CORSMiddleware
import io
import joblib
import torch
import torch.nn as nn
import pandas as pd
from pydantic import BaseModel, Field
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms


app = FastAPI(
    title="Flipkart Order Intelligence API",
    version="1.0.0",
)
app.include_router(agent_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://nihit-jain.github.io",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# CNN
# --------------------------------------------------

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


# --------------------------------------------------
# Load models
# --------------------------------------------------

return_risk_model = joblib.load(
    "models/return_risk_model.pkl"
)

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

class OrderFeatures(BaseModel):
    product_category: str
    price_inr: float = Field(ge=0)
    discount_pct: float = Field(ge=0, le=100)
    payment_method: str
    customer_tenure_days: int = Field(ge=0)
    num_previous_orders: int = Field(ge=0)
    num_previous_returns: int = Field(ge=0)
    delivery_distance_km: float = Field(ge=0)
    delivery_days: int = Field(ge=0)
    is_weekend_order: int = Field(ge=0, le=1)
    rating_given: float = Field(ge=0, le=5)
# --------------------------------------------------
# Return-risk endpoint
# --------------------------------------------------

@app.post("/predict-return")
def predict_return(order: OrderFeatures):

    # Support both:
    # 1. Flat request
    # 2. {"order_features": {...}}

    if "order_features" in order:
        order = order["order_features"]

    order_df = pd.DataFrame([order.model_dump()])

    probability = return_risk_model.predict_proba(
        order_df
    )[0][1]

    prediction = int(probability >= 0.44)

    return {
        "return_probability": round(
            float(probability), 4
        ),
        "return_probability_percent": round(
            float(probability * 100), 2
        ),
        "prediction": (
            "Likely return"
            if prediction == 1
            else "Likely not return"
        ),
    }


# --------------------------------------------------
# Image prediction endpoint
# --------------------------------------------------

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...)
):
    contents = await file.read()

    suffix = (
        Path(file.filename).suffix
        if file.filename
        else ".png"
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        return classify_product_image(
            temp_path
        )

    finally:
        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )