import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from MyPytorchModel import MyPytorchModel

# dtype = torch.float
# device = torch.device("mps")



# settings

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

hparams = {
    "batch_size": 4,
    "lr": 0.001,
    "momentum": 0.9,
    "input_size": 3 * 32 * 32,
    "hidden_size": 210,
    "num_classes": 10,
    "reg": 1e-5
}



# loading data

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams["batch_size"], shuffle=True, num_workers=0)
print(len(trainset))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=hparams["batch_size"], shuffle=False, num_workers=0)#
print(len(testset))

classes = trainset.classes



# learning

net = MyPytorchModel(hparams)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=hparams["lr"], weight_decay=hparams["reg"])

for epoch in range(15):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999: 
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')



# preparing submission files

from utils import save_and_zip
save_and_zip(net)



# testing

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')