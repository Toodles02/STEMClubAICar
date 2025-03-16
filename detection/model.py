import torch.nn as nn 
import torch 
from data import process, human_label

class TrafficSignDetector(nn.Module):
    def __init__(self):
        super(TrafficSignDetector, self).__init__() 
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.linear_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.Linear(256, 30)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x 
    
best_model = 'models/model_98.pth'

def predict(image): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TrafficSignDetector().to(device)
    state_dict = torch.load(best_model)
    model.load_state_dict(state_dict)
    

    image = process(image).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output)
        return human_label(pred)

