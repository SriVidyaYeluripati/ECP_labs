import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch import nn, optim

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load and Preprocess CIFAR-100 Dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
])

# Download and load CIFAR-100 train and test datasets
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Load the state_dict
state_dict = torch.load("resnet18.pt")

# Initialize a ResNet18 model
model = models.resnet18(pretrained=False)

# Modify the first convolution layer (conv1) to match the saved model (3x3 kernel)
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

# Modify the fc layer to match CIFAR10 (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Load the state_dict into the model
model.load_state_dict(state_dict)

# Now modify the fc layer for CIFAR100 (100 classes)
num_classes = 100
in_features = model.fc.in_features  # Get the input features of the fc layer
model.fc = nn.Linear(in_features, num_classes)  # Replace the fc layer
model = model.to(device)  # Move the model to the same device as the inputs

# Step 3: Define Loss Function, Optimizer, and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Step 4: Training Function
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Step 5: Evaluation Function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Step 6: Train and Evaluate the Model
num_epochs = 100

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Step 7: Save the Model Weights
torch.save(model.state_dict(), 'resnet18cifar100.pt')
print("Model weights saved to resnet18cifar100.pt")

