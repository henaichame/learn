# -*- coding: utf-8 -*-

import shutil
import os
import glob
import re
import fnmatch
oldpath = r'F:\ASD\Package_1189312\eeg_sub_files01'
newpath = r'D:\Desktop\EEG436\EEG436TD'
file_path = r'D:\Desktop\EEG436\EEG436TD.txt'


# 从文件中获取要拷贝的文件的信息
def get_filename_from_txt(file):
    filename_lists = []
    # 编码信息有可能不一致，encoding ='utf-8' 改为文件对应编码
    with open(file, 'r', encoding='utf-8') as f:
        lists = f.readlines()
        for list in lists:
            filename_lists.append(str(list).strip('\n'))
    return filename_lists


# 拷贝文件到新的文件夹中
def mycopy(srcpath, dstpath, filename):
    if not os.path.exists(srcpath):
        print("srcpath not exist!")
    if not os.path.exists(dstpath):
        print("dstpath not exist!")
    for root, dirs, files in os.walk(srcpath, True):
        for filename1 in files:
            if fnmatch.fnmatch(filename1, filename + '*.mat'):
                # 如果存在就拷贝
                shutil.copy(os.path.join(root, filename1), dstpath)
            else:
                # 不存在的话将文件信息打印出来
                print(filename1)


if __name__ == "__main__":
    # 执行获取文件信息的程序
    filename_lists = get_filename_from_txt(file_path)
    # 根据获取的信息进行遍历输出
    for filename in filename_lists:
        mycopy(oldpath, newpath, filename)
