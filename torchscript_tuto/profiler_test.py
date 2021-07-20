import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_profiler import profile


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.conv1(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 44944)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@profile
def f():
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(10, 3, 224, 224)
    # example.to("cuda")

    net = Net()
    # net.to("cuda")
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net, example)


    print(traced_script_module.graph)
    print(traced_script_module.code)


    # traced_script_module.save("/home/sdat789/HONG/torchscript/torchscript_tuto/net.pt")
if __name__ == "__main__":
    f()
    #python -m memory_profiler <file_name>.py