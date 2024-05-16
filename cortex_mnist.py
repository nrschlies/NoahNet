import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(0)
np.random.seed(0)

class MNISTCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images = self.data.iloc[:, 1:].values.astype('uint8')
        self.labels = self.data.iloc[:, 0].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.images[index].reshape(28, 28)
        label = self.labels[index]
        image = Image.fromarray(image, mode='L')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNISTCSV('mnist_train.csv', transform=transform)
test_dataset = MNISTCSV('mnist_test.csv', transform=transform)

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    epochs = 15  # Reduced number of epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=True)
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed precision
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            loop.set_description(f"Epoch {epoch}/{epochs}")
            loop.set_postfix(loss=train_loss / (total / 128), accuracy=100. * correct / total)

        scheduler.step()  # Step the scheduler at the end of each epoch

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        loop = tqdm(test_loader, leave=True)
        with torch.no_grad():
            for data, target in loop:
                data, target = data.to(device), target.to(device)
                with torch.cuda.amp.autocast():  # Mixed precision
                    output = model(data)
                    loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                loop.set_postfix(loss=test_loss / (total / 128), accuracy=100. * correct / total)
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / total
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        print('\nClassification Report:\n', classification_report(all_targets, all_preds, digits=4))
        print('Confusion Matrix:\n', confusion_matrix(all_targets, all_preds))
