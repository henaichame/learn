import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from vgg_model import vgg
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {  # 数据预处理函数
    # transforms.Compose:将使用的预处理方法打包成一个整体
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪为224*224像素大小
                                 transforms.RandomHorizontalFlip(),  # 水平方向上随机翻转
                                 transforms.ToTensor(),
                                 # 将原本的取值范围0-255转化为0.1~1.0
                                 # 且将原来的序列（H，W，C）转化为（C，H，W）
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                 # 使用均值和标准差对Tensor进行标准化

    "val": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# image_path = data_root + "/data_set/flower_data/"  # flower data set path

root_path = os.getcwd()
data_path = os.path.abspath(os.path.join(root_path, "data_set"))
image_path = os.path.abspath(os.path.join(data_path, "flower_data"))
train_path = os.path.abspath(os.path.join(image_path, "train"))
val_path = os.path.abspath(os.path.join(image_path, "val"))
model_path = os.path.abspath(os.path.join(root_path, "model"))

train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
train_num = len(train_dataset)
flower_list = train_dataset.class_to_idx  # 类名对应的索引
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
cla_dict = dict((val, key) for key, val in flower_list.items())  # 遍历字典，将key val值返回
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 通过json将cla_dict字典进行编码
with open('class.json', 'w') as json_file:
    json_file.write(json_str)  # ‘class_indices.json’, 将字典的key值保存在文件中，方便在之后的预测中读取它的信息

"""
通过DataLoader将train_dataset载入进来
shuffle=True, 随机参数，随机从样本中获取数据
num_workers=0 加载数据所使用的线程个数，线程增多，加快数据的生成速度，也能提高训练的速度，
在Windows环境下线程数为0，在Linux系统下可以设置为多线程，一般设置为8或者16
"""
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
validate_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

model_name = "vgg16"
# net = vgg(model_name=model_name, num_classes=6, init_weights=True)
net = vgg(model_name=model_name)
# num_class根据自己的数据集修改，类别数+1(空的背景)，init_weights=True表示初始化权重
net.to(device)  # 连接到设备
loss_function = nn.CrossEntropyLoss()  # 损失函数，使用的是针对多类别的损失交叉熵函数
optimizer = optim.Adam(net.parameters(), lr=0.0001)   # 优化器，优化对象为网络所有的可训练参数，学习率设置为0.0001

best_acc = 0.0  # 定义最佳准确率，在训练过程中保存准确率最高的一次训练的模型
# save_path = './{}Net.pth'.format(model_name)
save_path = os.path.join(model_path, '{}Net.pth'.format(model_name))  # 保存权重的路径

for epoch in range(30):  # 设置epoch的轮数
    # train
    """
    net.train():启用dropout，net.eval():关闭dropout
    Dropout:随机失活神将元，防止过拟合
    但是只希望在训练的过程中如此，并不希望在预测的过程中起作用
    通过net.train()和net.eval()来管理Dropout方法和BN方法
    """
    net.train()
    running_loss = 0.0  # 用于统计训练过程中的平均损失
    for step, data in enumerate(train_loader, start=0):  # 遍历数据集，返回每一批数据data以及data对应的step
        images, labels = data  # 将数据分为图像和标签
        optimizer.zero_grad()  # 清空之前的梯度信息（清空历史损失梯度）
        outputs = net(images.to(device))  # 将输入的图片引入到网络，将训练图像指认到一个设备中，进行正向传播得到输出，
        loss = loss_function(outputs, labels.to(device))  # 将网络预测的值与真实的标签值进行对比，计算损失梯度
        loss.backward()  # 误差的反向传播
        optimizer.step()  # 通过优化器对每个结点的参数进行更新

        # print statistics
        running_loss += loss.item()  # 将每次计算的loss累加到running_loss中
        # print train process
        rate = (step + 1) / len(train_loader)  # 计算训练进度，当前的步数除以训练一轮所需要的总的步数
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        # 打印训练进度
    print()

    # validation
    net.eval()  # 关闭Dropout
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():  # torch.no_grad()之后的计算过程中，不计算每个结点的误差损失梯度
        for val_data in validate_loader:
            """
            遍历验证集；将数据划分为图片和对应的标签值
            对结点进行参数的更新；指认到设备上并传入网络得到输出
            """
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            """求得输出的最大值作为预测最有可能的类别"""
            predict_y = torch.max(outputs, dim=1)[1]
            """
            predict_y == val_labels.to(device) 预测类别与真实标签值的对比，相同为1，不同为0
            item()获取数据，将Tensor转换为数值，通过sum()加到acc中，求和可得预测正确的样本数
            """
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        """如果准确率大于历史最优的准确率，将当前值赋值给最优，并且保存当前的权重"""
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))  # 打印训练到第几轮，累加的平均误差，以及最优的准确率

print('Finished Training')