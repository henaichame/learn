import os
import numpy as np
import scipy.io as scio
# import seaborn as sns
# import matplotlib.pyplot as plt
import time
# 记录运行时间
time.process_time()


# filepath 指定处理的文件路径，例如"F:\EEG421_pearson\ASD_BM\bm_001"
# type_dir 控制数据存储在哪个文件夹
# def process_data(filepath, type_dir):
#     # type_dir = r'F:\EEG421_pearson_ASD'
#     # 获取当前路径下的文件名，返回List
#     pathDir = os.listdir(filepath)
#     # a = os.path.dirname(filepath)
#
#     # 创建存储数组
#     pearson_alpha = []
#     pearson_beta = []
#     pearson_delta = []
#     pearson_gamma1 = []
#     pearson_gamma2 = []
#     pearson_theta = []
#     pearson_all = []  # 表示未分频段的数据
#     pearson_sum = []  # 表示分了频段之后，将所有数据融合在一起
#
#     for file in pathDir:
#         # newDir = os.path.join(filepath, file)  # 将文件命加入到当前文件路径后面
#         if os.path.splitext(file)[1] == '.mat':
#
#             # 计算频段融合数据
#             sum_filename = os.path.join(filepath, file)
#             sum_mat_data = scio.loadmat(sum_filename)
#             sum_data_raw = sum_mat_data['data_pearson']
#             sum_data_ing = np.mat(sum_data_raw)[np.triu_indices(125, 1)]
#             sum_data_ing = np.array(sum_data_ing)
#             pearson_sum.extend(sum_data_ing)
#
#             # 分别计算各频段数据
#             if 'alpha' in file:  # 获取alpha数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_alpha.extend(data_ing)
#             elif 'beta' in file:  # 获取beta数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_beta.extend(data_ing)
#             elif 'delta' in file:  # 获取delta数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_delta.extend(data_ing)
#             elif 'gamma1' in file:  # 获取gamma1(30 -- 60 hz)数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_gamma1.extend(data_ing)
#             elif 'gamma2' in file:  # 获取gamma2(30 -- 100 hz)数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_gamma2.extend(data_ing)
#             elif 'theta' in file:  # 获取theta数据
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_theta.extend(data_ing)
#             else:  # 未分频段信号
#                 filename = os.path.join(filepath, file)
#                 mat_data = scio.loadmat(filename)
#                 data_raw = mat_data['data_pearson']
#                 data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
#                 data_ing = np.array(data_ing)
#                 pearson_all.extend(data_ing)
#     #  获取数据所属名称，例如rest_001
#     filedir_name = os.path.basename(filepath)
#
#     #  分别保存每个个体的各频段以及融合数据
#     pearson_alpha = np.array(pearson_alpha)
#     rename_alpha = filedir_name + '_pearson_alpha.mat'
#     '''
#     如果想要每人一个文件夹存放数据需要这样操作
#     dir_alpha = os.path.join('F:\\EEG421_pearson_matrix', filedir_name)
#     file_alpha = os.path.join('F:\\EEG421_pearson_matrix', filedir_name, rename_alpha)
#     os.makedirs(dir_alpha, exist_ok=True)
#     '''
#     #  所有人放在同一个文件夹中，训练模型时通过名字辨别
#
#     file_alpha = os.path.join(type_dir, rename_alpha)
#     scio.savemat(file_alpha, {'pearson_alpha': pearson_alpha})
#
#     pearson_beta = np.array(pearson_beta)
#     rename_beta = filedir_name + '_pearson_beta.mat'
#     file_beta = os.path.join(type_dir, rename_beta)
#     scio.savemat(file_beta, {'pearson_beta': pearson_beta})
#
#     pearson_delta = np.array(pearson_delta)
#     rename_delta = filedir_name + '_pearson_delta.mat'
#     file_delta = os.path.join(type_dir, rename_delta)
#     scio.savemat(file_delta, {'pearson_delta': pearson_delta})
#
#     pearson_gamma1 = np.array(pearson_gamma1)
#     rename_gamma1 = filedir_name + '_pearson_gamma1.mat'
#     file_gamma1 = os.path.join(type_dir, rename_gamma1)
#     scio.savemat(file_gamma1, {'pearson_gamma1': pearson_gamma1})
#
#     pearson_gamma2 = np.array(pearson_gamma2)
#     rename_gamma2 = filedir_name + '_pearson_gamma2.mat'
#     file_gamma2 = os.path.join(type_dir, rename_gamma2)
#     scio.savemat(file_gamma2, {'pearson_gamma2': pearson_gamma2})
#
#     pearson_theta = np.array(pearson_theta)
#     rename_theta = filedir_name + '_pearson_theta.mat'
#     file_theta = os.path.join(type_dir, rename_theta)
#     scio.savemat(file_theta, {'pearson_theta': pearson_theta})
#
#     pearson_all = np.array(pearson_all)
#     rename_all = filedir_name + '_pearson_all.mat'
#     file_all = os.path.join(type_dir, rename_all)
#     scio.savemat(file_all, {'pearson_all': pearson_all})
#
#     pearson_sum = np.array(pearson_sum)
#     rename_sum = filedir_name + '_pearson_sum.mat'
#     file_sum = os.path.join(type_dir, rename_sum)
#     scio.savemat(file_sum, {'pearson_sum': pearson_sum})
#
def process_data(filepath, type_dir):
    # type_dir = r'F:\EEG421_pearson_ASD'
    # 获取当前路径下的文件名，返回List
    pathDir = os.listdir(filepath)
    # a = os.path.dirname(filepath)

    # 创建存储数组
    pearson_alpha = []
    pearson_beta = []
    pearson_delta = []
    pearson_gamma1 = []
    pearson_gamma2 = []
    pearson_theta = []
    pearson_all = []  # 表示未分频段的数据
    pearson_sum = []  # 表示分了频段之后，将所有数据融合在一起

    for file in pathDir:
        # newDir = os.path.join(filepath, file)  # 将文件命加入到当前文件路径后面
        if os.path.splitext(file)[1] == '.mat':

            data_description = 'data_mi'
            # 计算频段融合数据
            sum_filename = os.path.join(filepath, file)
            sum_mat_data = scio.loadmat(sum_filename)
            sum_data_raw = sum_mat_data[data_description]
            sum_data_ing = np.mat(sum_data_raw)[np.triu_indices(125, 1)]
            sum_data_ing = np.array(sum_data_ing)
            pearson_sum.extend(sum_data_ing)

            # 分别计算各频段数据
            if 'alpha' in file:  # 获取alpha数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_alpha.extend(data_ing)
            elif 'beta' in file:  # 获取beta数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_beta.extend(data_ing)
            elif 'delta' in file:  # 获取delta数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_delta.extend(data_ing)
            elif 'gamma1' in file:  # 获取gamma1(30 -- 60 hz)数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_gamma1.extend(data_ing)
            elif 'gamma2' in file:  # 获取gamma2(30 -- 100 hz)数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_gamma2.extend(data_ing)
            elif 'theta' in file:  # 获取theta数据
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_theta.extend(data_ing)
            else:  # 未分频段信号
                filename = os.path.join(filepath, file)
                mat_data = scio.loadmat(filename)
                data_raw = mat_data[data_description]
                data_ing = np.mat(data_raw)[np.triu_indices(125, 1)]
                data_ing = np.array(data_ing)
                pearson_all.extend(data_ing)
    #  获取数据所属名称，例如rest_001
    filedir_name = os.path.basename(filepath)

    #  分别保存每个个体的各频段以及融合数据
    pearson_alpha = np.array(pearson_alpha)
    rename_alpha = filedir_name + '_mi_alpha.mat'
    '''
    如果想要每人一个文件夹存放数据需要这样操作
    dir_alpha = os.path.join('F:\\EEG421_pearson_matrix', filedir_name)
    file_alpha = os.path.join('F:\\EEG421_pearson_matrix', filedir_name, rename_alpha)
    os.makedirs(dir_alpha, exist_ok=True)
    '''
    #  所有人放在同一个文件夹中，训练模型时通过名字辨别

    file_alpha = os.path.join(type_dir, rename_alpha)
    scio.savemat(file_alpha, {'mi_alpha': pearson_alpha})

    pearson_beta = np.array(pearson_beta)
    rename_beta = filedir_name + '_mi_beta.mat'
    file_beta = os.path.join(type_dir, rename_beta)
    scio.savemat(file_beta, {'mi_beta': pearson_beta})

    pearson_delta = np.array(pearson_delta)
    rename_delta = filedir_name + '_mi_delta.mat'
    file_delta = os.path.join(type_dir, rename_delta)
    scio.savemat(file_delta, {'mi_delta': pearson_delta})

    pearson_gamma1 = np.array(pearson_gamma1)
    rename_gamma1 = filedir_name + '_mi_gamma1.mat'
    file_gamma1 = os.path.join(type_dir, rename_gamma1)
    scio.savemat(file_gamma1, {'mi_gamma1': pearson_gamma1})

    pearson_gamma2 = np.array(pearson_gamma2)
    rename_gamma2 = filedir_name + '_mi_gamma2.mat'
    file_gamma2 = os.path.join(type_dir, rename_gamma2)
    scio.savemat(file_gamma2, {'mi_gamma2': pearson_gamma2})

    pearson_theta = np.array(pearson_theta)
    rename_theta = filedir_name + '_mi_theta.mat'
    file_theta = os.path.join(type_dir, rename_theta)
    scio.savemat(file_theta, {'mi_theta': pearson_theta})

    pearson_all = np.array(pearson_all)
    rename_all = filedir_name + '_mi_all.mat'
    file_all = os.path.join(type_dir, rename_all)
    scio.savemat(file_all, {'mi_all': pearson_all})

    pearson_sum = np.array(pearson_sum)
    rename_sum = filedir_name + '_mi_sum.mat'
    file_sum = os.path.join(type_dir, rename_sum)
    scio.savemat(file_sum, {'mi_sum': pearson_sum})


def traversal_dir(dirpath):
    # 遍历目录，组成每个rest_文件夹路径
    # 外层循环遍历 6 个文件夹，内层循环遍历6个文件夹中的小文件夹目录
    for dirs in os.listdir(dirpath):
        dir_path = os.path.join(dirpath, dirs)
        for dirss in os.listdir(dir_path):
            dir_paths = os.path.join(dir_path, dirss)
            process_data(dir_paths, r'F:\EEG421_mi_ASD')


traversal_dir(r'F:\EEG421_mi_a')
# process_data(r'F:\EEG421_mi_t\TD_REST\rest_071', r'F:\EEG421_mi_TD')
# 记录运行时间
print("运行时间是: {:5.5}s".format(time.process_time()))
