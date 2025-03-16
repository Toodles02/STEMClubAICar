import torch 
import torch.nn as nn
import time 
import matplotlib.pyplot as plt 
from model import TrafficSignDetector
from data import load_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def test(model_path): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TrafficSignDetector().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)  

    criterion = nn.CrossEntropyLoss().to(device) 
    model.eval()  

    print("Loaded model")

    _, _, test_loader = load_data() 

    correct = 0 
    loss = 0
    total = 0 
    start = time.time() 
    
    total_labels = []
    total_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            result = criterion(outputs, labels)
            loss += result.item() 
            
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum().item() 
            total += labels.size(0)

            total_labels.extend(labels.cpu().numpy())
            total_pred.extend(pred.cpu().numpy())

    cm = confusion_matrix(total_labels, total_pred)
    display = ConfusionMatrixDisplay(cm)
    display.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for model at {model_path}")
    plt.savefig('cm/confusion.png')
    plt.close()

    print("Generated confusion matrix")

    elapsed = time.time() - start 
    accuracy = (correct / total) * 100
    avg_loss = loss / len(test_loader)

    print(f"Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}, Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    test('models/model_98.pth')