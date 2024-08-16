import time
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import torchsummary
from tqdm import tqdm
import gc
import psutil
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
# from sklearn.metrics.pairwise import cosine_similarity 
from model import MNN, BasicBlock, TaskNet
from plot import plot_loss, plot_heat

process = psutil.Process(os.getpid())


# Training
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    total = len(train_loader)

    for epoch in range(num_epochs):

        for index, (images, labels) in tqdm(enumerate(train_loader), total=total, desc=f'Epoch [{epoch}/{num_epochs}] : Learning ', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            # if (index + 1) % 100 == 0:
            #     print(f'[{index+1} /{total}] Loss: {loss.item()}')

        # # save model
        # torch.save(model.state_dict(), os.path.join(
        #     OUT_DIR, f"MNN_MNIST_{tid}.pth"))

        # model.load_state_dict(torch.load(f"MN_MNIST_{epoch+1}.pth"))

        # save model
        # torch.save(model.state_dict(), f"MN_CIFAR10_{epoch+1}.pth")

        # model.load_state_dict(torch.load(f"MN_CIFAR10_{epoch+1}.pth"))

    return model


# testing
def evaluate_model(model, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return np.round(accuracy, 5)


num = 0


# Evaluate old tasks
def evaluate_old_task(model: MNN, tasks_test_loader: dict, new_tid):
    accs = []
    global num
    num += 1

    for tid, test_loader in tasks_test_loader.items():
        if tid > new_tid:
            continue

        tm = model.get_task_model(tid)
        path = os.path.join(OUT_DIR, f"task[{tid}]-{num}.pth")
        torch.save(tm.state_dict(), path)
        # accuracy = evaluate_model(tm, test_loader)
        # print(f'评估任务[{tid}]:{accuracy}')
        # accuracy = evaluate_model(tm, test_loader)
        # print(f'评估任务[{tid}]:{accuracy}')
        accuracy = evaluate_model(tm, test_loader)
        print(f'评估任务[{tid}]:{accuracy}')
        accs.append(accuracy)

    return accs


# Calculate task similarity
def cal_sim(model, loader):

    sim_matrix = []

    # Extract features for new tasks
    for tid, task_features in model.Task_Feature.items():
        tm = model.get_task_model(tid)
        features = extract_task_feature(tm, loader)
        # print(
        #     f"tid:{tid}, task_features:{len(task_features)}, features:{len(features)}", )

        sims = []
        for i, feature in enumerate(features):
            tf = task_features[i]
            val = compare_features(tf, feature)
            sims.append(val)

        sim_matrix.append(sims)

        # 释放内存
        del tm
        del features

    sim_matrix = np.array(sim_matrix)

    if sim_matrix.shape[0] == 0:
        sim_matrix = np.array([[0.0 for _ in range(model.num_layers)]])

    return sim_matrix


# 提取任务特征
def extract_task_feature(model: TaskNet, train_loader):
    features = []
    conf = []

    def forward_hook(module, input, output):
        # print(module, output.shape)
        f = output.cpu().detach().numpy()
        f = np.mean(f, axis=(0))
        features.append(f)

    handles = []
    for layer in model.layers.children():
        handle = layer.register_forward_hook(forward_hook)
        handles.append(handle)

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            features = []

            model(images)

            # assert False, "Expected a single output from the model"
            conf.append(features)
            # break

    for handle in handles:
        handle.remove()

    res = []
    for i in range(len(model.layers)):
        arr = [fs[i] for fs in conf]

        images = np.mean(arr, 0)
        # print(f"Layer[{i}] {images.shape=}")

        num, h, w = images.shape
        # 计算要将图片排列为接近方形的网格大小 Calculate to arrange the images into a grid size close to a square
        grid_size = int(np.ceil(np.sqrt(num)))
        grid_total = grid_size ** 2

        # 填充空白图片，使得图片总数为 grid_size^2  Fill in blank image, Make the total number of pictures is grid_size^2
        if grid_total > len(images):
            padding = np.zeros((grid_total - num, h, w))
            images = np.concatenate((images, padding), axis=0)

        # 重新调整图片数组的形状为 Reshape the image array to (grid_size, grid_size, h, w)
        images = images.reshape(grid_size, grid_size, h, w)

        # 将每一行的图片拼接在一起 Horizontal stacking of each row of images together
        rows = [np.hstack(row) for row in images]

        # 将所有行再垂直拼接起来 Vertically stacked array
        final_image = np.vstack(rows)
        # print(f"{final_image=}")

        # Remove the abnormal value
        array = remove_outliers_zscore(final_image.flatten(), threshold=3)

        # 归一化到[0, 1]范围  Normalize to the range of [0,1]
        array = (final_image - array.min()) / (array.max() - array.min())
        # print(array[:5, :5])
        array = array * 255
        array = array.astype(np.uint8)

        res.append(array)
    return res


def remove_outliers_zscore(data, threshold=3):
    # calculate Z-score
    z_scores = stats.zscore(data)

    # Abnormal value detection （Filter the data point of the Z-SCORE absolute value greater than the threshold）
    filtered_data = [x for x, z in zip(
        data, z_scores) if np.abs(z) < threshold]

    return np.array(filtered_data)


# 对比特征
def compare_features(features1, features2):
    # print("features1.shape", features1.shape)
    # print("features2.shape", features2.shape)

    # return ssim(features1, features2)
    return  consine_similarity(features1, features2)
 

def consine_similarity(features1, features2):
    # Flatten high-dimensional arrays into one-dimensional vectors
    v1 = features1.flatten()
    v2 = features2.flatten()
    # # Calculate the dot product of vectors
    # dot_product = np.dot(v1, v2)
    # # 计算向量的范数
    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
    # 计算余弦相似度
    # similarity = dot_product / (norm_v1 * norm_v2)
    # res=np.linalg.norm(features1 - features2)
    # print("cosine_similarity", cosine_similarity)
    
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(torch.Tensor(v1), torch.Tensor(v2))
    return similarity

#define evaluation metric: backward transfer BWT
def cal_BWT(task_accs: list, tid):
    a = []
    for accs in task_accs.values():
        a.append(accs[-1])
    a = np.array(task_accs[tid])-np.array(a)
    return np.mean(a[:-1])

#define evaluation metric: forgetting measure FM
def cal_FM(task_accs: list, tid):
    max_len = np.max([len(accs) for accs in task_accs.values()])
    a = [accs + [0]*(max_len-len(accs)) for accs in task_accs.values()]
    a = np.array(a)

    np.savetxt(os.path.join(OUT_DIR, "task_accs.txt"), a, fmt="%.5f")
    x = np.max(a[:tid, :tid], axis=0)
    y = a[tid, :tid]
    return np.mean(x-y)


# Load dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# root = '/data/qyan439/DataSet/CIFAR10'
# train_dataset = datasets.CIFAR10(
#     root=root, train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(
#     root=root, train=False, download=True, transform=transform)
# num_classes=10
# init_channels=3
# task_classes = [(0,2),(1,2),(3,4),(4,5),(6,7,),(8,9,)]


root = '/data/qyan439/DataSet/CIFAR100'
train_dataset = datasets.CIFAR100(
    root=root, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(
    root=root, train=False, download=True, transform=transform)
num_classes=100
init_channels=3
task_classes = [list(range(i*10, i*10+10)) for i in range(0, 10)]
#task_classes = [list(range(i*2, i*2+2)) for i in range(0, 50)]

# root = '/data/qyan439/DataSet/'
# train_dataset = datasets.MNIST(
#     root=root, train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(
#     root=root, train=False, download=True, transform=transform)
# num_classes=10
# init_channels=1
# task_classes = [(0,2),(1,2),(3,4),(4,5),(6,7,),(8,9,)]


batch_size = 128

# test_loader = DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False)


def split_dataset2task(dataset: Dataset, task_classes: list[tuple]):
    targets = np.array(dataset.targets)
    groups = {}
    # 遍历dataset，将每个图片的索引添加到对应标签的列表中
    for index, label in enumerate(targets):
        if label not in groups:
            groups[label] = []
        groups[label].append(index)
    print("Classes:", np.array(list(groups.keys())))

    task = {}
    for tid, classes in enumerate(task_classes):
        index_list = np.concatenate([groups[i] for i in classes], axis=0)
        # print(f"index_list", index_list)
        ss = Subset(dataset, indices=index_list)
        task[tid] = ss
        print(f"Task {tid} has {classes} Classes about {len(index_list)} Images")

        # torch.save(ss, f'./mnist_{label}.pth')
    return task


num_task = len(task_classes)
print("Number of tasks:", num_task)

tasks_train_dataset = split_dataset2task(train_dataset,  task_classes)

tasks_test_dataset = split_dataset2task(test_dataset,  task_classes)

tasks_train_loader = {}
for tid, ds in tasks_train_dataset.items():
    tasks_train_loader[tid] = DataLoader(ds, batch_size, shuffle=True)

tasks_test_loader = {}
for tid, ds in tasks_test_dataset.items():
    tasks_test_loader[tid] = DataLoader(ds, batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

# Initialize the model, loss function, and optimizer
layer_channels = [64, 128, 256, 512]
threshold = [0.8, 0.90, 0.95, 0.99]
model = MNN(BasicBlock, [2, 2, 2, 2], layer_channels,
            num_classes=num_classes, init_channels=3, threshold=threshold)

# torchsummary.summary(model, (1, 28, 28))


criterion = nn.CrossEntropyLoss()
learn_rate = 0.01

task_accs = {}

train_losses = []
parent_dir = os.path.dirname(__file__)
OUT_DIR = os.path.join(parent_dir, 'out')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print(f"{OUT_DIR=}")


AAs = []
AIAs = []
FMs = []
BWTs = []

total_paras = 0

num_epochs = 10

print("\nStart !\n")

# Training and evaluation model
for tid, train_loader in tasks_train_loader.items():

    sim_matrix = cal_sim(model, train_loader)
    print(f"sim_matrix: {sim_matrix.shape}\n{np.round(sim_matrix, 5)}\n")
    # Save similarity matrix

    np.savetxt(os.path.join(OUT_DIR, "sim_matrix.txt"), sim_matrix, fmt='%.5f')

    model.new_task(tid, sim_matrix)
    # print("model.network:", model.network)
    model.to(device)
    # torchsummary.summary(model.network, (1, 28, 28))

    optimizer = optim.Adam(model.network.parameters(), lr=learn_rate)

    ################################
    # print(f'Task [{tid}/{num_task}]  ')
    start_time = time.time()

    train_model(model, train_loader, criterion, optimizer, num_epochs)

    end_time = time.time()
    tt = end_time - start_time
    print(f'Task [{tid}/{num_task}] finished in {tt:.2f} seconds\n')

    ################################
    # test
    # pgs = [p.requires_grad for p in model.network.parameters()]
    # print(f"{pgs=}")
    p0 = model.network.layers[-1].parameters()
    p0 = next(p0).cpu().detach().numpy().reshape(-1, 32)[:512]
    path = os.path.join(OUT_DIR,  f"task[{tid}]-layer[3]-p0.txt")
    np.savetxt(path, p0, fmt='%.2f')

    # Extract current task features
    feats = extract_task_feature(model.network, train_loader)
    model.Task_Feature[tid] = feats
    print(f"Num of feature for Task[{tid}] : {len(model.Task_Feature[tid])}")

    for layer, array in enumerate(feats):

        # 将NumPy数组转换为Pillow图像对象
        image = Image.fromarray(array)

        # 保存为图片
        path = os.path.join(OUT_DIR, f"Task[{tid}]-Layer[{layer}].png")
        image.save(path)

    # Evaluate the model on the test set
    # test_loader = tasks_test_dataset[tid]
    # accuracy = evaluate_model(model, test_loader)
    # print(f'Test Accuracy: {accuracy:.2f}%')

    # Evaluate old tasks
    accs = evaluate_old_task(model, tasks_test_loader, tid)
    print(f"accs for each task after learning task[{tid}]:", np.round(accs, 2))
    task_accs[tid] = accs

    AAs.append(np.mean(accs))
    aia = np.mean(AAs)
    AIAs.append(aia)
    print(f"AAs: {np.round(AAs,2)} \nAIAs: {np.round(AIAs,2)}")

    if tid > 0:
        # 评估FM
        fm = cal_FM(task_accs, tid)
        FMs.append(fm)
        # BWT
        bwt = cal_BWT(task_accs, tid)
        BWTs.append(bwt)
        print(f"FMs: {np.round(FMs,2)} \nBWTs: {np.round(BWTs,2)} ")

    # 计算模块重用率
    count = 0
    for layer in model.Layer_Block:
        count += len(layer)

    total = len(model.Task_Block_Map.keys())*len(model.Layer_Block)
    model.reuse_rate = (total-count) / total*100
    print(f"Module reuse rate: {model.reuse_rate:.2f}%")

    # 任务模块关系
    print("Task Block Map:", model.Task_Block_Map)
    tbm = list(model.Task_Block_Map.values())
    path = os.path.join(OUT_DIR, "Task_Block_Map.txt")
    np.savetxt(path,  tbm, fmt='%d')

    # 模型大小

    for layer in model.Layer_Block:
        n1 = 0
        for p in layer[0].parameters():
            n1 += p.nelement()
        total_paras += n1*len(layer)

    for fl in model.Task_FL.values():
        for b in fl:
            for p in b.parameters():
                total_paras += p.nelement()

    print(f"Number of Model Parameters: {total_paras:,}")

    # 计算内存使用情况
    # from pympler import summary, muppy

    # # 获取所有对象的摘要
    # all_objects = muppy.get_objects()
    # summary.print_(summary.summarize(all_objects))

    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")  # Resident Set Size
    print(f"VMS: {memory_info.vms / 1024 ** 2:.2f} MB")  # Virtual Memory Size

    print("\n===================================================\n")

    # 调用垃圾回收
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()

    # 清空 CUDA 缓存
    torch.cuda.empty_cache()


print("All tasks finished.")


#path = os.path.join(OUT_DIR, 'Task_Epoch_Loss.png')
#plot_loss(train_losses, path)


#path = os.path.join(OUT_DIR, 'Reuse_Block_Map.png')
#array = np.loadtxt(os.path.join(OUT_DIR, 'Task_Block_Map.txt'), dtype=np.uint8)
#plot_heat(array, path)
