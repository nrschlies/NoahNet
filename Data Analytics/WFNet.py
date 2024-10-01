import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class WaveWeightLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(WaveWeightLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.A = nn.Parameter(torch.Tensor(out_features, in_features))
        self.f = nn.Parameter(torch.Tensor(out_features, in_features))
        self.phi = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.f, -np.pi, np.pi)
        nn.init.uniform_(self.phi, -np.pi, np.pi)
        nn.init.uniform_(self.b, -0.1, 0.1)
        nn.init.zeros_(self.B)
        
    def forward(self, x):
        w = self.A * torch.sin(self.f * x.unsqueeze(1) + self.phi) + self.b
        out = torch.sum(w * x.unsqueeze(1), dim=2) + self.B
        return out

class WFNet(nn.Module):
    def __init__(self):
        super(WFNet, self).__init__()
        self.flatten = nn.Flatten()
        self.wave1 = WaveWeightLayer(784, 300)
        self.ln1 = nn.LayerNorm(300)
        self.dropout1 = nn.Dropout(0.2)
        
        self.wave2 = WaveWeightLayer(300, 100)
        self.ln2 = nn.LayerNorm(100)
        self.dropout2 = nn.Dropout(0.2)
        
        self.wave3 = WaveWeightLayer(100, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.wave1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.wave2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.wave3(x)
        
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    print(f"Epoch {epoch} - Training started")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Print progress for every 10% of the epoch completed
        if (batch_idx + 1) % (total_batches // 10) == 0:
            percent_complete = 100 * (batch_idx + 1) / total_batches
            print(f'Epoch {epoch} - {percent_complete:.1f}% complete')

    print(f"Epoch {epoch} - Training completed")
    return total_loss / len(train_loader)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_batches = len(test_loader)
    print(f"Epoch {epoch} - Testing started")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Print progress for every 10% of the test completed
            if (batch_idx + 1) % (total_batches // 10) == 0:
                percent_complete = 100 * (batch_idx + 1) / total_batches
                print(f'Epoch {epoch} - Testing {percent_complete:.1f}% complete')

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Epoch {epoch} - Testing completed")
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformation for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = WFNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader, epoch)
        
        print(f'Epoch {epoch} - Results:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.2f}%')

    print('Final evaluation:')
    final_test_loss, final_accuracy = test(model, device, test_loader, num_epochs)
    print(f'Final Test Loss: {final_test_loss:.4f}')
    print(f'Final Accuracy: {final_accuracy:.2f}%')

if __name__ == '__main__':
    main()
