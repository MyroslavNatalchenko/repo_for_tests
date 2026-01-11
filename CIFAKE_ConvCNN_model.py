import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class CIFAKE_ConvCNN(nn.Module):
    def __init__(self):
        super(CIFAKE_ConvCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32*8*8, 64)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x))) # 16*16
        x = self.maxpool2(self.relu2(self.conv2(x))) # 8*8
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def get_data_loaders(root, batch_size=64):
    training_data = datasets.ImageFolder(root=os.path.join(root, 'train'),
                                         transform=ToTensor())
    test_data = datasets.ImageFolder(root=os.path.join(root, 'test'),
                                         transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train_and_evaluate(model, train_loader, val_loader, epochs=10, device='mps'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=correct_train / total_train)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)

            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print(f"Time taken for training: {time.time() - start_time:.2f}s")
    return history


def get_metrics_and_plot(model, test_loader, history, device='mps'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Precision:\t{precision:.4f}")
    print(f"Recall:\t{recall:.4f}")
    print(f"F1 Score:\t{f1:.4f}")

    plt.figure(figsize=(30, 10))

    #Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    #Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    #Confusion Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('training_results/conv_cnn/conv_cnn.png')
    plt.show()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    train_loader, test_loader = get_data_loaders('dataset', 64)
    model = CIFAKE_ConvCNN().to(DEVICE)

    history = train_and_evaluate(model, train_loader, test_loader, 10, DEVICE)

    get_metrics_and_plot(model, test_loader, history, DEVICE)

    torch.save(model.state_dict(), 'training_results/conv_cnn/conv_cnn.pth')