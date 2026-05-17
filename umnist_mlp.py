
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")


def flatten_data(dataset):
    X = []
    y = []
    for img, label in dataset:
        X.append(img.view(-1).numpy())
        y.append(label)
    
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

import numpy as np

X_train_full, y_train_full = flatten_data(train_dataset)
X_test, y_test = flatten_data(test_dataset)


val_size = 10000
train_size = len(X_train_full) - val_size

X_train = X_train_full[:train_size]
y_train = y_train_full[:train_size]
X_val = X_train_full[train_size:]
y_val = y_train_full[train_size:]

print(f"\nTrain: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")


class MNIST_MLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, num_classes=10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.01, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        # TRAINING
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # VALIDATION
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val.numpy(), val_preds.numpy())
            val_losses.append(val_loss.item())
            val_accs.append(val_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs


print("\n" + "="*60)
print("BASELINE MODEL (Hidden: 128→64, Learning Rate: 0.01)")
print("="*60)

model_baseline = MNIST_MLP()
train_losses_baseline, val_losses_baseline, train_accs_baseline, val_accs_baseline = train_model(
    model_baseline, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.01, verbose=True
)

# تقييم baseline
model_baseline.eval()
with torch.no_grad():
    test_outputs = model_baseline(X_test)
    test_loss = nn.CrossEntropyLoss()(test_outputs, y_test)
    _, test_preds = torch.max(test_outputs, 1)
    test_acc_baseline = accuracy_score(y_test.numpy(), test_preds.numpy())

print(f"\n Baseline Test Accuracy: {test_acc_baseline:.4f} ({test_acc_baseline*100:.2f}%)")
print(f"   Test Loss: {test_loss.item():.4f}")


print("\n" + "="*60)
print("EXPERIMENT 1: Different Hidden Layer Sizes (64 → 32)")
print("="*60)

class MNIST_MLP_Exp1(nn.Module):
    def __init__(self, input_size=784, hidden1_size=64, hidden2_size=32, num_classes=10):
        super(MNIST_MLP_Exp1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_exp1 = MNIST_MLP_Exp1()
train_losses_exp1, val_losses_exp1, train_accs_exp1, val_accs_exp1 = train_model(
    model_exp1, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.01, verbose=True
)

model_exp1.eval()
with torch.no_grad():
    test_outputs_exp1 = model_exp1(X_test)
    _, test_preds_exp1 = torch.max(test_outputs_exp1, 1)
    test_acc_exp1 = accuracy_score(y_test.numpy(), test_preds_exp1.numpy())

print(f"\n Experiment 1 Test Accuracy: {test_acc_exp1:.4f} ({test_acc_exp1*100:.2f}%)")


print("\n" + "="*60)
print("EXPERIMENT 2: Different Learning Rate (LR = 0.005)")
print("="*60)

model_exp2 = MNIST_MLP()  # Same architecture as baseline
train_losses_exp2, val_losses_exp2, train_accs_exp2, val_accs_exp2 = train_model(
    model_exp2, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.005, verbose=True
)

model_exp2.eval()
with torch.no_grad():
    test_outputs_exp2 = model_exp2(X_test)
    _, test_preds_exp2 = torch.max(test_outputs_exp2, 1)
    test_acc_exp2 = accuracy_score(y_test.numpy(), test_preds_exp2.numpy())

print(f"\n Experiment 2 (LR=0.005) Test Accuracy: {test_acc_exp2:.4f} ({test_acc_exp2*100:.2f}%)")


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.plot(train_losses_baseline, label='Train Loss (Baseline)', color='blue')
plt.plot(val_losses_baseline, label='Val Loss (Baseline)', color='red', linestyle='dashed')
plt.plot(train_losses_exp1, label='Train Loss (Exp1)', color='green')
plt.plot(val_losses_exp1, label='Val Loss (Exp1)', color='orange', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison: Baseline vs Exp1')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_losses_baseline, label='Train Loss (Baseline)', color='blue')
plt.plot(val_losses_baseline, label='Val Loss (Baseline)', color='red', linestyle='dashed')
plt.plot(train_losses_exp2, label='Train Loss (Exp2)', color='green')
plt.plot(val_losses_exp2, label='Val Loss (Exp2)', color='orange', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison: Baseline vs Exp2 (LR=0.005)')
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 3)
plt.plot(val_accs_baseline, label='Val Acc (Baseline)', color='blue')
plt.plot(val_accs_exp1, label='Val Acc (Exp1)', color='green')
plt.plot(val_accs_exp2, label='Val Acc (Exp2)', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("RESULTS COMPARISON TABLE")
print("="*60)
print(f"{'Experiment':<35} {'Test Accuracy':<15} {'Difference':<10}")
print("-" * 60)
print(f"{'Baseline (128→64, LR=0.01)':<35} {test_acc_baseline:.4f} ({test_acc_baseline*100:.2f}%) {'---':<10}")
print(f"{'Exp 1 (64→32, LR=0.01)':<35} {test_acc_exp1:.4f} ({test_acc_exp1*100:.2f}%) {test_acc_exp1 - test_acc_baseline:+.4f}")
print(f"{'Exp 2 (128→64, LR=0.005)':<35} {test_acc_exp2:.4f} ({test_acc_exp2*100:.2f}%) {test_acc_exp2 - test_acc_baseline:+.4f}")

print("\n" + "="*60)
print("FINDINGS:")
print("="*60)
print(f"✓ Baseline achieved {test_acc_baseline*100:.2f}% accuracy")
print(f"✓ Experiment 1 (smaller network) achieved {test_acc_exp1*100:.2f}%")
print(f"✓ Experiment 2 (LR=0.005) achieved {test_acc_exp2*100:.2f}%")


print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print("✓ Problem: Handwritten Digit Recognition (MNIST)")
print("✓ Model: Multilayer Perceptron (Input → Hidden1 → Hidden2 → Output)")
print("✓ Input Layer: 784 neurons (28x28 pixels)")
print("✓ Hidden Layer 1: 128 neurons (Baseline)")
print("✓ Hidden Layer 2: 64 neurons (Baseline)")
print("✓ Output Layer: 10 neurons (digits 0-9)")
print("✓ Activation Functions: ReLU")
print("✓ Loss Function: CrossEntropyLoss")
print("✓ Optimizer: SGD")
print("✓ Experiments: 2 (Hidden layer sizes, Learning rate)")
print("✓ Visualizations: Loss & Accuracy curves")
print("="*60)