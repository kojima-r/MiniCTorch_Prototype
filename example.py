import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import minictorch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)


def main():
    model = Net()
    input_to_model = torch.randn((1,2))
    print(input_to_model.shape)
    model.eval()
    with torch.no_grad():
        filename="sample.json"
        print("[SAVE]",filename)
        minictorch.generator.generate_minictorch_file(model, input_to_model, filename)


if __name__ == "__main__":
    main()
