import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet101(pretrained=True)
# print(model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)
# print(example)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.

# with torch.jit.optimized_execution(model):
traced_script_module = torch.jit.trace(model, example, check_trace=True)
print(torch.jit._get_trace_graph(model,example))
# script_module = torch.jit.script(model, example)
# print(traced_script_module.code)


output = traced_script_module(torch.ones(1, 3, 224, 224))
# print(output)

# traced_script_module.save("/home/sdat789/HONG/torchscript/torchscript_tuto/traced_resnet101_model.pt")