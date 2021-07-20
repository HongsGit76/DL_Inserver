import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms, datasets, models
import torch.fx

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(44944, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(660*110, 110)

    def forward(self, x):
        # x = self.conv1(x)

        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 44944)
        x = x.view(-1, 660*110)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
        

#     def forward(self, x):
#         x = self.fc1(x)
#         return x


# An example input you would normally provide to your model's forward() method.
lin_ex = torch.rand(10)
example = torch.rand(1, 3, 224, 224)
check_ex = [torch.rand(1,3,224,224),torch.rand(1,3,224,224)]
# example.to("cuda")

net = Net()
# torch.save(net, "/home/sdat789/HONG/torchscript/torchscript_tuto/net_not_traced.pt")
# net.to("cuda")
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# print(net.state_dict().)
# print("-------------------------------------------------------------------------------------------------------------")
# print("input")
# print(example.size())
# print("output")
# print(net(example).size())
# print("--------------------------------------------------------------------------------------------------------------")
# print(torch.jit._get_trace_graph(net, example,_force_outplace=True, return_inputs=True, _return_inputs_states=True))
# print(torch.jit._get_trace_graph(net, example))
# print(list(net.parameters()))

# optim = torch.optim.SGD(net.parameters(), lr=0.01)
# traced_script_module = torch.jit.trace_module(traced_script_module,example)
# with  torch.jit.optimized_execution(True):
traced_script_module = torch.jit.trace(net, example)
tsm = torch.jit.script(net,example)
# print(traced_script_module.graph)
# out = traced_script_module.forward(example)
# print(out)

# print(traced_script_module)
# print("__________________________")
print(tsm.graph)
# tsm.save("/home/sdat789/HONG/torchscript/torchscript_tuto/f.pt")

# print(traced_script_module.graph)
# print(traced_script_module.code)

# traced_script_module.save("/home/sdat789/HONG/torchscript/torchscript_tuto/optim_net.pt")

# criterion = optim.SGD(net.parameters(), 0.01, 0.1)
# ans = torch.arange(10, dtype=torch.long)

# with profile(activities=[ProfilerActivity.CPU],
#         profile_memory=True, record_shapes=True) as prof:
#     out = net(example)
#     # criterion.zero_grad()
#     loss = F.nll_loss(out, ans)
#     loss.backward()
    

# print(prof.key_averages().table())

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     net(example)

# prof.export_chrome_trace("trace.json")



# print(resnet50)
# trace_module = torch.jit.trace(resnet50, example)
# print(trace_module.graph)
# print(torch.jit._get_trace_graph(resnet50, example))
# x = torch.jit._get_trace_graph(resnet50, example)
# print(x[0].__dir__())
# print(x[0])
# st = torch.fx.symbolic_trace(resnet18,)
# st = torch.fx.symbolic_trace(net)
# print(st)