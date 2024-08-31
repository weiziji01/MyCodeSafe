# 将isprs比赛数据集按照类别划分
import shutil

txt_path = r'/mnt/d/data/isprs_game/train_set/gt.txt'
img_path = r'/mnt/d/data/isprs_game/train_set/image/'
i0_path = r'/mnt/d/data/isprs_game/train_set/0/'
i1_path = r'/mnt/d/data/isprs_game/train_set/1/'

imgs = []
with open(txt_path, 'r') as f:
    for line in f:
        line = line.strip('\n')
        labels = line.split()
        imgs.append((labels[0], labels[1]))

for i in range(len(imgs)):
    if int(imgs[i][1]) == 0:
        source_img = img_path+imgs[i][0]
        shutil.copy(source_img, i0_path)
    if int(imgs[i][1]) == 1:
        source_img = img_path+imgs[i][0]
        shutil.copy(source_img, i1_path)
    i=i+1