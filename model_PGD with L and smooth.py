#!/usr/bin/env python3
import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 32


class Net(nn.Module):

    model_file = "models/mymodel.pth"

    def __init__(self):
        super().__init__()
        # Apply Spectral Normalization for Lipschitz Regularization
        self.conv1 = spectral_norm(nn.Conv2d(3, 6, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = spectral_norm(nn.Conv2d(6, 16, 5))
        self.fc1 = spectral_norm(nn.Linear(16 * 5 * 5, 120))
        self.fc2 = spectral_norm(nn.Linear(120, 84))
        self.fc3 = spectral_norm(nn.Linear(84, 10))
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, Net.model_file))


def add_random_noise(inputs, noise_std=0.1):
    noise = torch.randn_like(inputs) * noise_std
    return torch.clamp(inputs + noise, 0, 1)


def get_train_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    return torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)


def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    return torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)


def pgd_attack(net, images, labels, epsilon=0.03, alpha=0.01, num_iter=40):
    net.train()
    perturbed_images = images.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        perturbed_images.requires_grad = True
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.clone().detach().requires_grad_(True)

    return perturbed_images


def test_pgd(net, test_loader, epsilon=0.03, alpha=0.01, num_iter=40):
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        perturbed_images = pgd_attack(net, images, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def adversarial_training_with_smoothing(net, train_loader, pth_filename, num_epochs, epsilon=0.03, alpha=0.01, num_iter=10, noise_std=0.1):
    print("Starting adversarial training with Lipschitz Regularization and Randomized Smoothing")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Add randomized smoothing noise
            smoothed_inputs = add_random_noise(inputs, noise_std)

            # Generate adversarial examples
            adv_inputs = pgd_attack(net, smoothed_inputs, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

            # Compute loss
            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.3f}")
                running_loss = 0.0

    net.save(pth_filename)
    print(f"Model saved in {pth_filename}")


def test_natural_with_smoothing(net, test_loader, noise_std=0.1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            smoothed_images = add_random_noise(images, noise_std)
            outputs = net(smoothed_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file, help="File to load/store model weights.")
    parser.add_argument('-f', '--force-train', action="store_true", help="Force training even if model exists.")
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    net = Net()
    net.to(device)

    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        train_transform = transforms.Compose([transforms.ToTensor()])
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        adversarial_training_with_smoothing(net, train_loader, args.model_file, num_epochs=args.num_epochs,
                                            epsilon=0.03, alpha=0.01, num_iter=40, noise_std=0.1)

    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)
    acc_nat = test_natural_with_smoothing(net, valid_loader, noise_std=0.1)
    print(f"Natural accuracy with smoothing: {acc_nat:.2f}%")
    acc_pgd = test_pgd(net, valid_loader, epsilon=0.03, alpha=0.01, num_iter=40)
    print(f"PGD attack accuracy: {acc_pgd:.2f}%")


if __name__ == "__main__":
    main()
