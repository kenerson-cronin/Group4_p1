from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
class Pipeline:
    def __init__(self):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])  # Normalize for grayscale images

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.lossFunc = nn.CrossEntropyLoss()

    def train_step(self, model, optimizer):
        model.train()
        epochloss = 0
        for batchcount, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # labels = torch.eye(len(classes))[labels].to(device)
            # pdb.set_trace()
            
            optimizer.zero_grad()

            y = model(images)

            loss = self.lossFunc(y, labels)     
            loss.backward()

            optimizer.step()
            
            epochloss += loss.item()

        return epochloss

    def val_step(self, model):
        correct = 0
        total = 0
        model.eval()
        """confusion = {
            pred:{actual: 0 for actual in self.classes} for pred in self.classes
        }"""
        confusion = np.zeros((len(self.classes),len(self.classes)))
        with torch.no_grad():
            for batchcount, (images, labels) in enumerate(self.testloader):

                images = images.to(self.device)
                labels = labels.to(self.device)

                y = model(images)

                _, predicted = torch.max(y, 1)
                if total < 1:
                    print("predicted,labels: ", type(predicted),type(labels) , " shapes: ", predicted.shape,labels.shape)
                for num in range(0,len(labels)):
                    confusion[predicted[num]][labels[num]] += 1
                    total += 1
                

                correct += (predicted == labels).sum().item()

        return correct*100/total


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

LOSS_FN = nn.CrossEntropyLoss()
