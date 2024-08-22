import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_mlp(model, criterion, optimizer, X_train, y_train, num_epochs=10000):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print(
                f"🌟 Epoch [{epoch + 1}/{num_epochs}]\n"
                f"🕒 Loss: {loss.item():.4f}\n"
                f"{'='*50}"
            )

def evaluate_mlp(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
        accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
        print(
            f"✅ Accuracy for MLP classifier: {accuracy:.2f}%\n"
            f"{'='*50}"
        )

        cm = confusion_matrix(y_test.cpu(), predicted.cpu())
        print(
            f"📊 Confusion matrix for MLP classifier:\n"
            f"{cm}\n"
            f"{'='*50}"
        )