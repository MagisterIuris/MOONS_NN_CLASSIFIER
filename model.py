import torch.nn as nn

class MOONSNet(nn.Module):
    def __init__(self):
        super(MOONSNet, self).__init__()
        self.layer1 = nn.Linear(2, 6)
        self.layer2 = nn.Linear(6, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x