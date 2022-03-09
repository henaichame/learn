import torch
from vgg_model import vgg
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

"""打包数据预处理"""
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("1.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = vgg(model_name="vgg16", num_classes=6)  # 初始化网络
# load model weights
model_weight_path = "./model/vgg16Net.pth"  # 导入权重参数
model.load_state_dict(torch.load(model_weight_path))  # 载入网络模型
model.eval()  # 关闭dropout
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)  # 经过softmax函数将输出变为概率分布
    predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处所对应的索引值
print(class_indict[str(predict_cla)],predict[predict_cla].numpy())  # 输出预测的类别名称以及预测的准确率
plt.show()