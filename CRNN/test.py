"""
Testing Script for CRNN Number Plate Recognition
Visualizes predictions on test images
"""

import os
import torch
import matplotlib.pyplot as plt
import string
from torchvision import transforms
from PIL import Image

from model import CRNN

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char: idx + 1 for idx, char in enumerate(CHARS)}
int_to_char = {idx: char for char, idx in char_to_int.items()}
num_classes = len(CHARS) + 1

# Decode CTC output logits to readable text
def ctc_decode(logits, int_to_char):
    max_probs = torch.argmax(logits, dim=2)
    decoded_strings = []

    for seq in max_probs:
        prev = -1
        decoded = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                decoded.append(int_to_char[idx])
            prev = idx
        decoded_strings.append("".join(decoded))

    return decoded_strings


test_transforms = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.42, 0.43, 0.41),
        std=(0.32, 0.32, 0.32)
    )
])


# Test the model on images and visualize predictions
def test_model(model, test_folder, test_transforms, model_path, int_to_char, num_images=8, device="cuda"):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_list = [os.path.join(test_folder, img) for img in os.listdir(test_folder)]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(min(num_images, len(image_list))):
        image = Image.open(image_list[i]).convert("RGB")
        img_display = image.copy()

        img_tensor = test_transforms(image)\
                   .unsqueeze(0)\
                    .to(device)

        with torch.no_grad():
            output = model(img_tensor)  

        pred = ctc_decode(output, int_to_char)[0]

        axes[i].imshow(img_display)
        axes[i].axis("off")
        axes[i].set_title(f"Pred: {pred}")

    plt.tight_layout()
    plt.show()

# Path to the test images folder
test_folder = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(H=100, W=200, num_classes=num_classes).to(device)

test_model(model=model, test_folder=test_folder, test_transforms=test_transforms, model_path="best_model.pth", int_to_char=int_to_char, num_images=8, device=device)