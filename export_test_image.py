from torchvision import datasets
from PIL import Image


dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
)

image, label = dataset[0]

image.save("fashion_test.png")

print("Saved: fashion_test.png")
print("Label:", label)