# 随机生成比赛数据集的标签，txt格式，每张图像一个，里面只有一个随机目标的标注
import os
import random
 
# 设置要操作的文件夹路径
folder_path = "/mnt/d/data/fair1m1/train/random-ann/"

# 创建指定数量个txt文件
for i in range(4742):
    i_str=str(i+1)
    # file_name='image-'+i_str+'.txt'
    file_name=i_str+'.txt'
    os.mknod(folder_path+file_name)
 
# 获取文件夹中所有的txt文件
txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
category=['vehicle','ship','airplane','other']
 
# 遍历每个txt文件并写入数据
for file_name in txt_files:
    with open(os.path.join(folder_path, file_name), 'a') as f:
        # 生成8个随机数并写入文件
        random_numbers = [random.randint(0, 3) for i in range(8)]
        
        f.write(' '.join(str(x) for x in random_numbers))
        f.write(' ship 0\n')