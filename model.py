import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        x = self.linear_relu_stack(x)
        #x = torch.sigmoid(x)
        #x = x.squeeze()
        return x

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

class RegressionModelHyper1(nn.Module):
    def __init__(self, hyper=0):
        super().__init__()
        if hyper == 1:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1))
        elif hyper == 2:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1))
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

class RegressionModelHyper2(nn.Module):
    def __init__(self, hyper=0):
        super().__init__()
        if hyper == 1:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1))
        elif hyper == 2:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1))
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(13, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = ClassificationModel().to(device)
    print(model)
    print(model.parameters())
    