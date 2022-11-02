import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 512),
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)
    print(model.parameters())
    