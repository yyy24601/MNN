import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        # 确保每次卷积算法都是确定的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 设置随机种子
# set_seed(2024)

def out_paras(model, layer):
    for index, param in enumerate(model[layer].parameters()):
        print(f"Layer[{layer}]:")

        print(f"Parameter[{index}]:")
        print(param)  # 打印第二层的参数
        print(f"Parameter[{index}].grad:")
        print(param.grad)


# Define a simple model
model = nn.Sequential(
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

# Freeze the parameters of Layer2
for param in model[2].parameters():
    param.requires_grad = False

out_paras(model, 2)
out_paras(model, 0)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Create some sample data
inputs = torch.randn(1, 3)
targets = torch.randn(1, 1)

# forward propogation
outputs = model(inputs)
loss = criterion(outputs, targets)

# backpropogation and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('==================')
# Check the parameters of the first layer(Layer0) update or not

out_paras(model, 2)
out_paras(model, 0)
