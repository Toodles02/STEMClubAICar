import torch
import torch.nn as nn 
import torch.optim as optim
from data import get_data 
from model import TrafficSignDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, valid_loader, _ =  get_data()

model = TrafficSignDetector().to(device)

lr = 1e-3

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters, lr=lr)

epochs = 20 


for e in range(epochs):
    model.train() 
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() 

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 
    
    model.eval() 
    correct = 0 
    total = 0 
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss = loss.item()

            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum().item() 
            total += labels.size(0)
    accuracy = (correct / total) * 100 
    print(f"Epoch: [{e}/{epochs}], Accuracy: {accuracy:.2f}%, Loss: {running_loss:.4f}")


torch.save(model.state_dict(), f"/models/model_{accuracy:.0f}.pth")
print("Model saved!")
