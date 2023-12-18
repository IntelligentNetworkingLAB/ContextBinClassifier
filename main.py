import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Read the H5 file
with h5py.File('put your file', 'r') as file:
    context_embeddings = file['context_embeddings'][:]
    ans_embeddings = file['ans_embeddings'][:]
    labels = file['labels'][:]

# Convert to PyTorch tensors
context_embeddings = torch.tensor(context_embeddings, dtype=torch.float32)
ans_embeddings = torch.tensor(ans_embeddings, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Split the data into training and testing sets
train_contexts, test_contexts, train_answers, test_answers, train_labels, test_labels = train_test_split(
    context_embeddings, ans_embeddings, labels, test_size=0.2, random_state=42)


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, contexts, answers, labels):
        self.contexts = contexts
        self.answers = answers
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.contexts[idx], self.answers[idx], self.labels[idx]


# Create datasets and dataloaders
train_dataset = CustomDataset(train_contexts, train_answers, train_labels)
test_dataset = CustomDataset(test_contexts, test_answers, test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class BinaryClassifier(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(context_embeddings.shape[1] + ans_embeddings.shape[1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 1)

    def forward(self, context, answer):
        x = torch.cat((context, answer), dim=1)
        x = F.leaky_relu(self.dropout1(self.fc1(x)))
        x = F.leaky_relu(self.dropout2(self.fc2(x)))
        x = F.leaky_relu(self.dropout3(self.fc3(x)))
        x = F.leaky_relu(self.dropout4(self.fc4(x)))
        x = F.leaky_relu(self.dropout5(self.fc5(x)))
        x = F.leaky_relu(self.dropout6(self.fc6(x)))
        x = F.leaky_relu(self.dropout7(self.fc7(x)))
        x = torch.sigmoid(self.fc8(x))
        return x

model = BinaryClassifier().to(device)


model = BinaryClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels, threshold=0.5):
    predicted = (outputs >= threshold).float()
    #predicted = outputs.round()  # Assuming binary classification (0 or 1)
    correct = (predicted == labels).float()  # Convert boolean values to floats
    accuracy = correct.sum() / len(correct)
    return accuracy

patience = 10  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to signify an improvement
best_val_loss = float('inf')
counter = 0  # Counts the number of epochs without improvement

for epoch in range(100):
    model.train()
    train_loss = 0
    train_accuracy = 0

    for contexts, answers, labels in train_loader:
        contexts, answers, labels = contexts.to(device), answers.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(contexts, answers)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += calculate_accuracy(outputs, labels.unsqueeze(1))

    # Average training loss and accuracy
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for contexts, answers, labels in test_loader:
            contexts, answers, labels = contexts.to(device), answers.to(device), labels.to(device)
            outputs = model(contexts, answers)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels.unsqueeze(1))

    # Average validation loss and accuracy
    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader)

    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Early Stopping Check
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0  # Reset counter if there's an improvement
        torch.save(model.state_dict(), 'best_bin_classifier.pth')  # Save the best model
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # Update the learning rate scheduler
    scheduler.step(val_loss)

# Save the model
torch.save(model.state_dict(), 'bin_classifier.pth')
