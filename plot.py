from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def plot_loss(train_losses, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Loss')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)

    # plt.show()


def plot_heat(array, path):

    layers = []

    for row in array:
        for layer, i in enumerate(row):
            if layer > len(layers)-1:
                layers.append([])
            if i > len(layers[layer])-1:
                layers[layer].append(1)
            else:
                layers[layer][i] += 1

    # 找到最长的子列表长度
    max_len = max(len(sublist) for sublist in layers)

    # 填充较短的子列表
    padded_list = [sublist + [0] * (max_len - len(sublist))
                   for sublist in layers]

    # 转换为 NumPy 数组
    data = np.array(padded_list)
    print("Reuse Block Map:\n", data)

    plt.figure()
    # 使用Seaborn绘制热图，并在每个单元格显示值
    sns.heatmap(data, annot=True, fmt="d", cmap='viridis')
    # plt.colorbar()
    plt.title('Reuse Block Map')
    plt.xlabel('Blocks')
    plt.ylabel('Layers')
    # 保存图表
    plt.savefig(path)
    # plt.show()
