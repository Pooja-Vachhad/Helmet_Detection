"""
CRNN (Convolutional Recurrent Neural Network) Architecture
Built from scratch for number plate recognition using CTC loss
"""

import torch.nn as nn

# CNN + RNN model 
class CRNN(nn.Module):
    def __init__(self, H, W, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * (H // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2) 
        B, T, C, H = x.size()
        x = x.reshape(B, T, C * H)

        x = self.fc(x)
        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x