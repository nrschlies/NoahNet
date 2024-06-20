import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# Define a more complex CNN
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # Placeholder, will be corrected dynamically
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor dynamically
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x, x.detach().cpu().numpy()

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Increase the batch size and use more workers to speed up data loading
    num_workers = 4 if device.type == 'cuda' else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    model = EnhancedCNN().to(device)
    
    # Dynamically set the number of input features for fc1 layer
    with torch.no_grad():
        sample_input = torch.zeros((1, 1, 28, 28)).to(device)
        sample_output = model.conv1(sample_input)
        sample_output = nn.ReLU()(sample_output)
        sample_output = nn.MaxPool2d(2)(sample_output)
        sample_output = model.conv2(sample_output)
        sample_output = nn.ReLU()(sample_output)
        sample_output = nn.MaxPool2d(2)(sample_output)
        num_features = sample_output.view(1, -1).size(1)
        model.fc1 = nn.Linear(num_features, 128).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store PCA results, loss, and accuracy for each epoch
    pca_results = []
    all_labels = []
    losses = []
    accuracies = []

    # Training the model and capturing PCA evolution
    num_epochs = 80
    capture_interval = 2
    num_interpolations = 10  # Define num_interpolations here

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs, features = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(trainloader)
        accuracy = 100. * correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if epoch % capture_interval == 0:
            # Collect features from the test set
            model.eval()
            epoch_features = []
            epoch_labels = []
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    _, features = model(images)
                    epoch_features.append(features)
                    epoch_labels.append(labels.cpu().numpy())
            
            epoch_features = np.vstack(epoch_features)
            epoch_labels = np.hstack(epoch_labels)
            
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(epoch_features)
            pca_results.append(pca_result)
            all_labels.append(epoch_labels)

    # Create an animation of PCA results
    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # Interpolation function to create smooth transitions between PCA results
    def interpolate_pca(pca_results, all_labels, num_interpolations=10):
        interpolated_pca = []
        interpolated_labels = []
        for i in range(len(pca_results) - 1):
            for j in range(num_interpolations):
                alpha = j / num_interpolations
                interpolated_pca.append((1 - alpha) * pca_results[i] + alpha * pca_results[i + 1])
                interpolated_labels.append(all_labels[i])
        interpolated_pca.append(pca_results[-1])
        interpolated_labels.append(all_labels[-1])
        return interpolated_pca, interpolated_labels

    interpolated_pca_results, interpolated_all_labels = interpolate_pca(pca_results, all_labels, num_interpolations)

    def update(frame):
        ax3d.clear()
        ax2d.clear()
        
        pca_result = interpolated_pca_results[frame]
        labels = interpolated_all_labels[frame]
        
        # 3D plot
        for i in range(10):
            indices = labels == i
            ax3d.scatter(pca_result[indices, 0], pca_result[indices, 1], pca_result[indices, 2], label=f'Digit {i}', alpha=0.5)
        ax3d.set_title(f'3D PCA of MNIST features at frame {frame}')
        ax3d.set_xlabel('PCA Component 1')
        ax3d.set_ylabel('PCA Component 2')
        ax3d.set_zlabel('PCA Component 3')
        ax3d.legend()
        ax3d.grid(True)
        # Rotate the camera
        ax3d.view_init(elev=20., azim=frame * (360 / len(interpolated_pca_results)))
        
        # 2D plot
        for i in range(10):
            indices = labels == i
            ax2d.scatter(pca_result[indices, 0], pca_result[indices, 1], label=f'Digit {i}', alpha=0.5)
        ax2d.set_title(f'2D PCA of MNIST features at frame {frame}')
        ax2d.set_xlabel('PCA Component 1')
        ax2d.set_ylabel('PCA Component 2')
        ax2d.legend()
        ax2d.grid(True)

        # Loss and Accuracy
        epoch_index = frame // num_interpolations
        ax2d.text(1.02, 1.02, f'Epoch Loss: {losses[epoch_index]:.4f}', transform=ax2d.transAxes, fontsize=12, verticalalignment='top')
        ax2d.text(1.02, 0.92, f'Epoch Accuracy: {accuracies[epoch_index]:.2f}%', transform=ax2d.transAxes, fontsize=12, verticalalignment='top')

    ani = FuncAnimation(fig, update, frames=len(interpolated_pca_results), repeat=True)

    # Save the animation
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("mnist_pca_evolution.mp4", writer=writer)

    plt.show()

if __name__ == '__main__':
    main()
