#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/mymodel.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
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
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))

class RandomizedEnsemble(nn.Module):
    def __init__(self, models):
        super(RandomizedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):

        random_idx = torch.randint(0, len(self.models), (1,)).item()
        chosen_model = self.models[random_idx]
        return chosen_model(x)
    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file, device="cpu"):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))
def train_ensemble(ensemble, train_loader, pth_filename, num_epochs):
    print("Starting training for randomized ensemble")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ensemble.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        ensemble.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            random_idx = torch.randint(0, len(ensemble.models), (1,)).item()
            chosen_model = ensemble.models[random_idx]

            optimizer.zero_grad()
            outputs = chosen_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    torch.save(ensemble.state_dict(), pth_filename)
    print('Randomized ensemble model saved in {}'.format(pth_filename))

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid
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
        # perturbed_images.requires_grad = True

    return perturbed_images
def test_pgd(net, test_loader, epsilon=0.03, alpha=0.01, num_iter=40):

    correct = 0
    total = 0
    
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        perturbed_images = pgd_attack(net, images, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

def fgsm_attack(net, images, labels, epsilon=0.03):
    images = images.clone().detach().to(device)
    images.requires_grad = True 
    outputs = net(images)
    loss = F.cross_entropy(outputs, labels)
    net.zero_grad()
    loss.backward()
    grad = images.grad.data
    perturbed_images = images + epsilon * grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images
def test_fgsm(net, test_loader, epsilon=0.03):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        perturbed_images = fgsm_attack(net, images, labels, epsilon=epsilon)
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

def pgd_linf_attack(net, images, labels, epsilon=0.03, alpha=0.01, num_iter=40):

    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True

    for _ in range(num_iter):
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images

def pgd_l2_attack(net, images, labels, epsilon=0.5, alpha=0.1, num_iter=40):
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True

    for _ in range(num_iter):
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
        grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)

        perturbed_images = perturbed_images + alpha * grad_normalized
        delta = perturbed_images - images
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
        delta = delta / torch.clamp(delta_norm / epsilon, min=1.0).view(-1, 1, 1, 1)
        perturbed_images = images + delta
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images
def mat_rand_training(net, train_loader, pth_filename, num_epochs, epsilon_linf=0.03, alpha_linf=0.01, 
                      epsilon_l2=0.5, alpha_l2=0.1, num_iter=40):
    print("Starting MAT-Rand training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if torch.rand(1).item() < 0.5:
                adv_inputs = pgd_linf_attack(net, inputs, labels, epsilon=epsilon_linf, alpha=alpha_linf, num_iter=num_iter)
            else:
                adv_inputs = pgd_l2_attack(net, inputs, labels, epsilon=epsilon_l2, alpha=alpha_l2, num_iter=num_iter)

            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def adversarial_training(net, train_loader, pth_filename, num_epochs, epsilon=0.03, alpha=0.01, num_iter=10):

    print("Starting adversarial training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            adv_inputs = pgd_attack(net, inputs, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0


    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))
def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)
    # models = [Net().to(device) for _ in range(5)]
    # net = RandomizedEnsemble(models).to(device)
    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        # train_model(net, train_loader, args.model_file, args.num_epochs)
        # train_ensemble(net, train_loader,args.model_file, num_epochs=args.num_epochs)
        # adversarial_training(net, train_loader, args.model_file, num_epochs=args.num_epochs,
        #                      epsilon=0.03, alpha=0.01, num_iter=40)
        mat_rand_training(net, train_loader,args.model_file, num_epochs=args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))
    # net=ensemble
    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)
    acc_pgd = test_pgd(net, valid_loader, epsilon=0.03, alpha=0.01, num_iter=40)
    print("Model accuracy under PGD attack (test): {}".format(acc_pgd))
    acc_fgsm = test_fgsm(net, valid_loader, epsilon=0.03)
    print("Model accuracy under FGSM attack (test): {}".format(acc_fgsm))
    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()