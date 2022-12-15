import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST

from model import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

# 加载数据

img_path = "test.jpg"
assert os.path.exists(img_path), "file: '{}' dode not exist.".format(img_path)

img = Image.open(img_path)

plt.show(img)
# [N,C,H,W]
img = data_transform(img)

# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# 读取类别字典
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

json_file = open(json_path, "r")
class_indict = json.load(json_file)

# 创建模型
model = AlexNet(classes=5).to(device)

# 加载训练好的模型权重
weights_path = "./AlexNet.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path))

model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print_res = "class:{} prob: {:.3}".format(class_indict[str(predict_cla)],
                                          predict[predict_cla].numpy())

plt.title(print_res)
for i in range(len(predict)):
    print("class:{:10}  prob:{:.3}".format(class_indict[str(i)],
                                           predict[i].numpy()))
plt.show()
