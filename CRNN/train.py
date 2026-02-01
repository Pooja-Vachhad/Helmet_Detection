import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from model import CRNN

# Put your folder path here
folder_path = ""

img_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 reserved for blank
int_to_char = {idx: char for char, idx in char_to_int.items()}
num_classes = len(CHARS) + 1  # +1 for CTC blank

labels = [os.path.splitext(img)[0] for img in os.listdir(folder_path)]

# Split data into training and validation sets
train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    img_paths, labels, test_size=0.2, random_state=42
)

# Image transformations for training data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.42, 0.43, 0.41), std=(0.32, 0.32, 0.32))
])

valid_transforms = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.42, 0.43, 0.41), std=(0.32, 0.32, 0.32))
])

# Custom dataset class for loading images and labels
class Custom(Dataset):
    def __init__(self, image_paths, labels, char_to_int, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_int = char_to_int
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        image = Image.open(image_file).convert('RGB') 

        if self.transforms:
            image = self.transforms(image) 

        label_str = self.labels[idx]
        label = torch.tensor([self.char_to_int[char] for char in label_str], dtype=torch.long)

        return image, label

# Custom collate function to handle variable-length labels for CTC loss
def ctc_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels_concat = torch.cat(labels, dim=0)

    return images, labels_concat, label_lengths

# Create datasets with split data
train_dataset = Custom(train_paths, train_labels, char_to_int, train_transforms)
valid_dataset = Custom(valid_paths, valid_labels, char_to_int, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=ctc_collate)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=ctc_collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(H=100, W=200, num_classes=num_classes).to(device)

# Train and validate the model with early stopping
def run_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=100, patience=10, device="cuda", output_file="best_model.pth"):
    train_losses = []
    valid_losses = []
    best_valid_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        total_train = 0

        for images, labels, label_lengths in train_loader:  
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(images)          
            outputs = outputs.permute(1, 0, 2)  

            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(
                outputs.log_softmax(2),
                labels,
                input_lengths,
                label_lengths
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        avg_train_loss = train_loss / total_train
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0
        total_valid = 0

        with torch.no_grad():
            for images, labels, label_lengths in valid_loader:  
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)

                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

                loss = criterion(
                    outputs.log_softmax(2),
                    labels,
                    input_lengths,
                    label_lengths
                )

                valid_loss += loss.item() * images.size(0)
                total_valid += images.size(0)

        avg_valid_loss = valid_loss / total_valid
        valid_losses.append(avg_valid_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Valid Loss: {avg_valid_loss:.4f}"
        )

        # Save best model and check for early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            counter = 0
            torch.save(model.state_dict(), output_file)
            print("Saved best model")
        else:
            counter += 1
            print(f"No improvement ({counter}/{patience})")

        if counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses, valid_losses


# Define loss function and optimizer
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9, momentum=0.9, weight_decay=1e-4)

# Start training
train_losses, valid_losses = run_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=100, patience=10, device=device, output_file="best_model.pth")