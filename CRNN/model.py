"""
CRNN (Convolutional Recurrent Neural Network) for number plate recongition.
Uses a pretrained ResNet-18 (through layer3) as the CNN feature extractor,
paired with a custom Bi-LSTM sequence head and CTC loss for variable-length alphanumeric text recongition.
"""

from torchvision import models
import torch.nn as nn

class CRNN(nn.Module):
   def __init__(self , H , W , num_classes):
     super().__init__()
     self.H = H
     self.W = W
     self.num_classes = num_classes

     resnet = models.resnet18(pretrained = True)
     self.cnn = nn.Sequential(*list(resnet.children())[:7])

     self.fc = nn.Sequential(
         nn.Linear(1792, 128),
         nn.ReLU(),
         nn.Dropout(0.5)
     )

     self.lstm  = nn.LSTM(
         input_size = 128,
         hidden_size = 128 ,
         num_layers = 2,
         batch_first = True ,
         bidirectional = True ,
         dropout = 0.3
     )

     self.classifier = nn.Linear(128 * 2 , num_classes)


   def forward(self , x):
      x = self.cnn(x)
      x = x.permute(0 , 3 , 1 , 2)
      B , T , C , H = x.size()
      x = x .reshape(B , T , C * H)
      x = self.fc(x)
      x , _ = self.lstm(x)
      x = self.classifier(x)

      return  x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(H = 100 , W = 200 , num_classes  = num_classes).to(device)
