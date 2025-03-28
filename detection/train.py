import torch
import torch.nn as nn 
import torch.optim as optim
import time 
import keyboard
from data import load_data
from model import TrafficSignDetector


def train(epochs, lr, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficSignDetector().to(device)
    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"Loaded previous model from {model_path}")

    train_loader, valid_loader, _ = load_data() 

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs with rate {lr}...")

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
        total_loss = 0
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
                total_loss += loss.item()

                _, pred = torch.max(outputs, dim=1)
                correct += (pred == labels).sum().item() 
                total += labels.size(0)
        elapsed = time.time() - start 
        accuracy = (correct / total) * 100 
        running_loss = total_loss / len(valid_loader)
        print(f"Epoch: [{e+1}/{epochs}], Accuracy: {accuracy:.2f}%, Loss: {running_loss:.4f}, Elapsed: {elapsed:.2f}s")


    torch.save(model.state_dict(), f"models/model_{accuracy:.0f}.pth")
    print("Model saved!")
    return accuracy

if __name__ == "__main__":
    train(10, 1e-6, "models/model_98.pth")
    