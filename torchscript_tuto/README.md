# abstract

파이썬으로 작성된 파이토치 모델을 스크립트화 하여 c++에 로딩

- lin.pt: sample input size => {10, 5}

```python
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(10, 5)
        def forward(self, x):
            x = self.fc1(x)
            return x
```

- net.pt: sample input size => {1,3,224,224}

```python
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
```

- exmaple-app.cpp

    - 로딩 후 포워딩 진행

    - 필요 파라미터 분석

    - c++ 에서 GPU로 model, data loading 후 관찰 => $nvprof --print-gpu-trace ./example-app net.pt


- profiler_test.py

    - 파이토치에서 제공하는 프로파일러