import os
from PIL import Image
import tqdm


def bmp2png(file_dir):
    for root, dirs, files in os.walk(file_dir):  # 获取所有文件
        # for file in files:  # 遍历所有文件名
        for idx, file in enumerate(tqdm.tqdm(files)):
            if os.path.splitext(file)[1] == ".bmp":  # 指定尾缀  ***重要***
                im = Image.open(os.path.join(root, file))  # open img file
                newname = file.split(".")[0] + ".jpg"  # new name for png file
                im.save(os.path.join(root, newname))  # 转为png


bmp2png("/mnt/d/HRSC/FullDataSet/AllImages")
