import torch
import torch.nn as nn
import torch.nn.functional as F

class MyPytorchModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["hidden_size"]),
            nn.LeakyReLU(),
            nn.Linear(hparams["hidden_size"],hparams["hidden_size"]),
            nn.LeakyReLU(),
            nn.Linear(hparams["hidden_size"],hparams["num_classes"])
        )

        self.hparams = hparams

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = x.view(x.shape[0], -1)
        x = self.model(x)

        return x