import os, sys

# 打开文件
path = "G:/EEG421/TD_WS/WS_0145/"

# dirs = os.listdir(path)

# 输出所有文件和文件夹
# for file in dirs:
#     if os.path.splitext(file)[1] == '.mat':
#         new = file[:12]
#         os.chdir(path)
#         os.rename(file, new)  # 进行重命名
#         print(file)

dirs = os.listdir(path)

n = 0
i = 0
for i in dirs[:-1]:
    # 设置旧文件名（就是路径+文件名）
    oldname = dirs[n]
    if n <= 9:
        # 设置新文件名
        newname = "ws_004_00" + str(n + 1) + '.mat'
    else:
        newname = "ws_004_0" + str(n + 1) + '.mat'
    # 用os模块中的rename方法对文件改名
    os.rename(path + oldname, path + newname)
    print(oldname, '======>', newname)

    n += 1

print(len(dirs))
