# 定义基本的残差块
import numpy as np
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out


# 定义可复用的残差块层
class TaskNet(nn.Module):
    def __init__(self, fb, layers, lb):
        super(TaskNet, self).__init__()
        self.fb = fb
        self.layers = layers
        self.lb = lb

    def forward(self, x):
        out = self.fb(x)
        out = self.layers(out)
        out = self.lb(out)
        return out


class MNN(nn.Module):

    def __init__(self, block, num_blocks, layer_channels, num_classes=10, init_channels=1, threshold=0.8):

        super(MNN, self).__init__()

        self.block = block
        self.num_blocks = num_blocks
        self.layer_channels = layer_channels
        self.num_classes = num_classes

        self.num_layers = len(layer_channels)
        if isinstance(threshold, list):
            self.threshold = threshold
        else:
            self.threshold = [threshold]*self.num_layers

        self.Task_FL = {}
        self.Layer_Block = [[] for _ in range(self.num_layers)]
        self.Task_Block_Map = {}
        self.Task_Feature = {}

        self.network = None
        self.cur_task = None
        self.init_channels = init_channels

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 获取可以重用的模块
            if b := self.get_reusable_blocks():
                b.parameters().require_grad_(False)
                layers.append(b)
            else:
                b = block(self.in_channels, out_channels, stride)
                # self.Block_List.append(b)
                layers.append(b)

            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def get_reusable_blocks(self):
        return None

    def get_task_feature(self, task_id):
        return self.Task_Feature.get(task_id, None)

    def get_task_model(self, task_id) -> TaskNet:
        fl = self.Task_FL[task_id]
        layers = []
        blocks = self.Task_Block_Map[task_id]
        for index_layer, index_block in enumerate(blocks):
            block = self.Layer_Block[index_layer][index_block]
            # 重置缓冲区
            # for buff in block.buffers():
            #     buff.zero_()
            layers.append(block)

        return TaskNet(fl[0], nn.Sequential(*layers), fl[-1])

    def set_task(self, task_id):
        # 根据任务ID设置网络
        if task_id in self.Task_Block_Map:

            self.network = self.get_task_model(task_id)
        else:
            raise ValueError(f"Task {task_id} not found in Task_Block_Map")

    def cal_feature(self, dataset, device):
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.network(inputs)
            if i == 0:
                feature = outputs.cpu().detach().numpy()
            else:
                feature = np.concatenate(
                    (feature, outputs.cpu().detach().numpy()), axis=0)

    def new_task(self, task_id, sim_matrix):
        # 创建新的任务

        index_max_sim = np.argmax(sim_matrix, axis=0)
        max_sim = np.max(sim_matrix, axis=0)

        self.in_channels = 64

        index_blocks = []
        blocks = []

        # 比对每个任务的特征，获取可以重用的模块

        for layer, sim_val in enumerate(max_sim):
            # 如果相似度大于0.8，则重用模块
            if sim_val > self.threshold[layer]:
                # 最相似模块的任务索引
                index_task = index_max_sim[layer]
                # 最相似模块的索引
                index_block_reusable = self.Task_Block_Map[index_task][layer]
                b = self.Layer_Block[layer][index_block_reusable]
                # 模拟_make_layer修改in_channels
                self.in_channels = self.layer_channels[layer]
                b.apply(freeze_parameters)
                blocks.append(b)
                index_blocks.append(index_block_reusable)

            else:
                # 如果相似度小于0.8，则创建新的模块
                new_block = self._make_layer(
                    self.block, self.layer_channels[layer], self.num_blocks[0], stride=2)

                # 添加新模块到当前层的模块列表中
                self.Layer_Block[layer].append(new_block)

                blocks.append(new_block)
                index_new_block = len(self.Layer_Block[layer]) - 1
                index_blocks.append(index_new_block)

        self.layers = nn.Sequential(*blocks)
        self.Task_Block_Map[task_id] = index_blocks

        self.first_block = nn.Sequential(nn.Conv2d(self.init_channels, 64, kernel_size=3,
                                                   stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.last_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(self.layer_channels[-1], self.num_classes))

        # 保存特定于任务id的网络模块
        self.Task_FL[task_id] = [self.first_block, self.last_block]

        del self.network
        self.network = TaskNet(self.first_block,
                               self.layers,
                               self.last_block)

    def _forward_impl_(self, x):
        # print("x.shape", x.shape)
        # if self.Task_Block_Map

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        # print('out', out.shape)
        out = self.fc(out)

        print(self.layer1.conv1.weight[0,0,0,0])

        return out

    def forward(self, x):
        # out = self._forward_impl_(x)
        out = self.network(x)
        return out


# 冻结模型
def freeze_parameters(model: nn.Module):
    for param in model.parameters():
        # print(param)  # 打印第一层的参数
        param.requires_grad = False

    for buff in model.buffers():
        buff.zero_()
