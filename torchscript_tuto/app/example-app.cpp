#include <torch/script.h> // One-stop header.
// #include <torch/csrc/jit/frontend/tracer.h>
#include <torch/torch.h> // 새로운 nn 정의

#include <iostream>
#include <memory>


struct Net : torch::nn::Module {
  Net(/*int64_t N, int64_t M*/)
      : linear(register_module("linear", torch::nn::Linear(10, 5/*N, M*/))) {
    another_bias = register_parameter("b", torch::randn(5/*M*/));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
    // GPU로 돌릴 떄 GPU 정의
    // torch::DeviceType device_type;
    // device_type = torch::kCUDA;
    // std::cout << "Cuda available, running on GPU" << std::endl;
    torch::jit::script::Module module;
    
    // try {
        
        // Net net();

        // auto net = std::make_shared<Net>();

        // for (const auto& p : net->parameters()) {
        //   std::cout << p << std::endl;
        // }
        
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        
        // GPU 사용시 모델 GPU로 전송
        // torch::Device device(device_type);
        // module.to(torch::Device(device));

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        // inputs.push_back(torch::ones({1,3,224,224})/*.to(torch::Device(device))*/);
        inputs.push_back(torch::randn({3,3,224,224})/*.to(torch::Device(device))*/);
        // inputs.push_back(torch::randn({1,3,224,224}));

        at::Tensor target = torch::randint(10,{3},torch::kLong);
        std::cout << target <<'\n';
        
        // std::cout << module.parameters().size() << '\n';

        // torch::optim::SGD optimizer(module.parameters());
        torch::optim::SGD optimizer(module.parameters(), 0.01);
        //SGD(torch::autograd::variable_list params, torch::optim::SGDOptions defaults)

        // optimizer.zero_grad();

        // Execute the model and turn its output into a tensor.
        for (size_t i = 0; i < 10; i++)
        {
          at::Tensor output = module.forward(inputs).toTensor();
          std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

          torch::Tensor loss = torch::nll_loss(output, target);

          std::cout << loss << '\n';

          loss.backward();
          module.set_optimized(true);
        }
        
        // at::Tensor output = module.forward(inputs).toTensor();
        // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

        // torch::Tensor loss = torch::nll_loss(output, target);

        // std::cout << loss << '\n';

        // loss.backward();

        // optimizer.step();

        // std::cout << loss.item<float>() << '\n';

        // for (const auto& p : module.parameters()){
        //   std::cout << p << std::endl;
        // }

        // std::cout << "-------------------" << '\n';

        // for (const auto& q : module.attributes()){
        //   std::cout << q << std::endl;
        // }

  // }
  // catch (const c10::Error& e) {
  //   std::cerr << "error loading the model\n";
  //   return -1;
  // }

  std::cout << "ok\n";
}