# coding:UTF-8
import scipy.io as scio

dataFile = r'G:\EEG421\ASD\NDARAB752FNL.Bm.mat'
# path = "G:/EEG421/TD_REST/"
mat_data = scio.loadmat(dataFile)
# print(type(mat_data))
# print(mat_data.keys())

# 获取key
key_name = list(mat_data.keys())[3]
print(key_name)
# 根据key获取数据
data = mat_data[key_name]
# 获取数据的shape,这里数据的shape是(1,25876)
# 后面表示的是图片的数量
print(data.shape)
# 遍历数据
# for line in data[0, :]:
#     # 获取图片的名称
#     img_name = line[1][0]
#     # 获取图片对应的标签信息
#     img_info = line[2][0]
