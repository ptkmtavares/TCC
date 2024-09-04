import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout com probabilidade de 50%
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def predict_mlp(model, X_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        return predicted, probabilities

def train_mlp(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=10000, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % (num_epochs // 4) == 0:
            print(
                f"🌟 Epoch [{epoch + 1}/{num_epochs}]\n"
                f"🕒 Loss: {loss.item():.4f}\n"
                f"🕒 Validation Loss: {val_loss.item():.4f}\n"
                f"{'='*75}"
            )

def evaluate_mlp(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
        accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
        #print(
        #    f"✅ Accuracy for MLP classifier: {accuracy:.2f}%\n"
        #    f"{'='*75}"
        #)

        cm = confusion_matrix(y_test.cpu(), predicted.cpu())
        print(
            f"Confusion matrix for MLP classifier:\n"
            f"{cm}\n"
            f"{'='*75}"
        )
        return accuracy