import json
import sys
import torch
from torch.optim import Adam
import os
import torch.nn as nn
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from model import AlexNet

# 定义要使用的设备 cpu或者gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("using {} device".format(device))

# 定义数据增强形式
data_transform = {
    "train":transforms.Compose([transforms.RandomResizedCrop(224),  # 随机尺寸修剪，看源码
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]),
    "val":transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  # ((mean)(std))
    ])
}
data_root = os.path.abspath("E:/deeplearn/代码/deep-learning-for-image-processing-master/")
image_path = os.path.join(data_root, "data_set") # flower data set path
assert os.path.exists(image_path), "{} path does not exist".format(image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 类别转换成字典形式{"classes1":0,"classes2":1,.......}
flower_list = train_dataset.class_to_idx

# 键值互换
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将键值互换后的数据写入到json文件
json_str = json.dumps(cla_dict, indent=4) # indent代表缩进级别
with open("class_indices.json","w") as json_file:
    json_file.write(json_str)

batch_size = 4

num_works = 0

# 加载数据集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])

val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False)

print("using {} images for training,{} image for vaildation".format(train_num, val_num))


net = AlexNet(classes=5, init_weights=True)
net.to(device)

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 定义优化器
optimizer = Adam(net.parameters(), lr=0.0002)

# 定义迭代次数
epochs = 10

save_path = "./AlexNet.pth"

best_acc = 0.0

train_steps = len(train_loader)

# 开始训练
for epoch in range(epochs):
    # train 训练模式,此时dropout会生效
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)

    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

    # val 验证模型 dropout不工作
    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print("[epoch %d] train_loss:%.3f val_accuracy:%.3f"% (epoch + 1, running_loss/train_steps, val_accurate))

        if val_accurate>best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)

    print("Finished Training")
