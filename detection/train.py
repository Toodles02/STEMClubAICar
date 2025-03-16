import torch
import torch.nn as nn 
import torch.optim as optim
import time 
import keyboard
from data import load_data
from model import TrafficSignDetector


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficSignDetector().to(device)

    train_loader, valid_loader, _ = load_data() 

    epochs = 1
    lr = 1e-3

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training...")

    exit_flag = False 

    accuracy = 0 
    for e in range(epochs):
        if exit_flag:
            break

        start = time.time() 

        model.train() 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if keyboard.is_pressed('q'): 
                print("Training interrupted, saving the model...")
                exit_flag = True 
                break 

            optimizer.zero_grad() 

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
        
        model.eval() 
        correct = 0 
        total = 0 
        if exit_flag:
            break 
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                if keyboard.is_pressed('q'): 
                    print("Training interrupted, saving the model...")
                    exit_flag = True 
                    break 

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss = loss.item()

                _, pred = torch.max(outputs, dim=1)
                correct += (pred == labels).sum().item() 
                total += labels.size(0)
        elapsed = time.time() - start 
        accuracy = (correct / total) * 100 
        print(f"Epoch: [{e+1}/{epochs}], Accuracy: {accuracy:.2f}%, Loss: {running_loss:.4f}, Elapsed: {elapsed:.2f}s")


    torch.save(model.state_dict(), f"models/model_{accuracy:.0f}.pth")
    print("Model saved!")

if __name__ == "__main__":
    train() 