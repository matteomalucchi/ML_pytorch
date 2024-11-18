import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Define the model
class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load dataset
def load_dataset():
    # Generate synthetic data for binary classification
    np.random.seed(42)
    X = np.random.rand(1000, 2)  # Features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels: 1 if sum > 1, else 0

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# Testing function
def test_model(model, test_dataloader):
    print("Testing the model on the test dataset...")
    return evaluate_model(model, test_dataloader)

# Plotting function
def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

# Main script
if __name__ == "__main__":
    dataset = load_dataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = 2
    hidden_size = 16
    output_size = 1

    model = SimpleDNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training the model...")
    loss_history = train_model(model, train_loader, criterion, optimizer, epochs=30)

    print("\nEvaluating the model on the validation dataset...")
    evaluate_model(model, val_loader)

    print("\nPlotting the training loss curve...")
    plot_loss_curve(loss_history)

    
