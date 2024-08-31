 
# * labelme标签转yolov8标签汇总
import os
import shutil
import json
import random

def deleteEmptyImg(rootdir, json_path):
    """
    删除没有标签的图片
    标签与图片需要在不同的文件夹中
    rootdir: 图片路径
    json_path: 标签文件存放的文件夹地址
    """
    n = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        # 遍历每一张图片
        for filename in filenames:
            currentPath = os.path.join(parent, filename)  # 目前图片的地址
            picture_number = filename[:-4]  # 图片去掉.png（.jpg）后的名字
            print("the picture name is:" + filename)
            json_file = os.listdir(json_path)  # 将标签文件存放在json_file列表里面
            if picture_number + '.json' in json_file:  # 如果图片有对应的标签文件则跳过
                pass
            else:
                print(currentPath)
                n += 1
                os.remove(currentPath)  # 没有对应的标签文件则删除
    print("删除的照片数为：", n)
    return 0


def mixFile(source_path, target_path):
    """
    将标签文件与目标文件混合到一个文件夹中
    source_path: 下面子目录包含图片文件夹与标签文件夹, 如images和label_json
    target_path: 将source_path及其子目录下的所有文件汇总到这个文件夹中
    """
    if not os.path.exists(target_path):     #目标文件夹不存在就新建
        os.makedirs(target_path)

    if os.path.exists(source_path):     #源文件夹存在才执行    
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                print(src_file)

    print('复制完成')
    return 0


def changeJsonImagePath(path):
    """
    将labelme生成的json格式的标签内的ImagePath进行修改，使其和图片对应起来
    （YOLO格式其实使根据标签文件名读取图片路径，但防止以后需要json标签，还是修改）
    图片的路径就是该函数修改后的路径
    path: 包含有图片和json标签的文件夹路径
    """
    dirs = os.listdir(path)
    
    num_flag = 0
    for file in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(file)[1] == ".json":  # 筛选Json文件
            num_flag = num_flag + 1
            print("path = ", file)                            # 此处file为json文件名，之前修改为与图片jpg同名
            # print(os.path.join(path,file))
            with open(os.path.join(path, file), 'r') as load_f: # 若有中文，可将r改为rb
                load_dict = json.load(load_f)                   # 用json.load()函数读取文件句柄，可以直接读取到这个文件中的所有内容，并且读取的结果返回为python的dict对象
            n = len(load_dict)  # 获取字典load_dict中list值
            print('n = ', n)
            print("imagePath = ", load_dict['imagePath'])                # 此处因为我的json文件要修改的imagePath， 没有那么多弯弯绕， 直接在顶层， 所以一层[]即可， 如果你们的不是这种结构， 需自行修改
    
    
            filename = file[:-5]                            # 去掉拓展名5位  .json
            print("filename = ", filename)
            load_dict['imagePath'] = filename + '.jpg'       # 存到当前路径下， 如果有其它存储要求， 自行修改即可
            print("new imagePath = ", load_dict['imagePath'])
    
            with open(os.path.join(path, file), 'w') as dump_f:
                json.dump(load_dict, dump_f)
    
    if (num_flag == 0):
        print('所选文件夹不存在json文件，请重新确认要选择的文件夹')
    else:
        print('共{}个json文件'.format(num_flag))
    return 0

 
def convertLabelmesToYolo(labelme_folder, yolotxt_folder, label_dict):
    """
    目标检测labelme标注数据转为yolov8需要的txt格式
    图片和标签需要在同一个文件夹中
    labelme_folder: 标注与图片混合后的文件夹
    yolotxt_folder: 存储yolo标准转换后的文件夹, 子目录是images与labels(txt格式文件夹)
    label_dict: yolo标注格式的字典
    """
    label_folder = os.path.join(yolotxt_folder, "labels")         # label_folder = image_folder/labels
    os.makedirs(label_folder, exist_ok=True)
    image_folder = os.path.join(yolotxt_folder, "images")         # image_folder = image_folder/images
    os.makedirs(image_folder, exist_ok=True)
 
    for root, dirs, files in os.walk(labelme_folder):          # root-文件夹所在路径  dir-路经下的文件夹   file-里面的文件
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[-1] != ".json":      # 分割文件名和扩展名
                shutil.copy(file_path, image_folder)
                print(f"Copied {file_path} to {image_folder}")
            else:
                # 读取json文件
                with open(file_path, 'r') as f:
                    labelme_data = json.load(f)
 
                image_filename = labelme_data["imagePath"]
                image_width = labelme_data["imageWidth"]
                image_height = labelme_data["imageHeight"]
 
                txt_filename = os.path.splitext(image_filename)[0] + ".txt"
                txt_path = os.path.join(label_folder, txt_filename)

                with open(txt_path, 'w') as f:
                    for shape in labelme_data["shapes"]:
                        label = shape["label"]
                        points = shape["points"]
                        x_min, y_min = points[0]
                        x_max, y_max = points[1]
 
                        center_x = round(((x_min + x_max) / 2) / image_width , 4)
                        center_y = round(((y_min + y_max) / 2) / image_height ,4)
                        width = round(abs(x_min - x_max) / image_width, 4)
                        height = round(abs(y_min - y_max) / image_height, 4)
 
                        class_id = label_dict[label]
                        f.write(f"{class_id}  {center_x:.6f} {center_y:.6f} {width} {height}\n")
 
                print(f"Converted {file} to {txt_path}")
 
    print("转换成功")
    return 0
 
def splitData(yolotxt_folder, split_folder, classes, idx_dict, split_rate):
    """
    将数据集分为训练集和验证集
    yolotxt_folder: 子目录包括图片(images)与yolo格式标注(labels, txt格式)的文件夹
    split_folder: 生成的最终文件, 可为yolo模型所用的, 子目录为train, val文件夹和data.yaml文件
        文件夹下还包括images和labels子目录
    """
    random.seed(0)
    # split_rate = 0.2 #验证集占比
    origin_label_path = os.path.join(yolotxt_folder, "labels")
    origin_image_path = os.path.join(yolotxt_folder, "images")
    train_label_path = os.path.join(split_folder, "train", "labels")
    os.makedirs(train_label_path, exist_ok=True)
    train_image_path = os.path.join(split_folder, "train", "images")
    os.makedirs(train_image_path, exist_ok=True)
    val_label_path = os.path.join(split_folder, "val", "labels")
    os.makedirs(val_label_path, exist_ok=True)
    val_image_path = os.path.join(split_folder, "val", "images")
    os.makedirs(val_image_path, exist_ok=True)
 
    images = os.listdir(origin_image_path)
    num = len(images)
    eval_index = random.sample(images,k=int(num*split_rate))
    for single_image in images:
        origin_single_image_path = os.path.join(origin_image_path, single_image)
        single_txt = os.path.splitext(single_image)[0] + ".txt"
        origin_single_txt_path = os.path.join(origin_label_path, single_txt)
        if single_image in eval_index:
            #single_json_path = os.path.join(val_label_path,single_json)
            shutil.copy(origin_single_image_path, val_image_path)
            shutil.copy(origin_single_txt_path, val_label_path)
        else:
            #single_json_path = os.path.join(train_label_path,single_json)
            shutil.copy(origin_single_image_path, train_image_path)
            shutil.copy(origin_single_txt_path, train_label_path)
 
    print("数据集划分完成")
 
    with open(os.path.join(split_folder,"data.yaml"),"w") as f:
        f.write(f"train: {split_folder}\n")
        f.write(f"val: {val_image_path}\n")
        f.write(f"test: {val_image_path}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    return 0



if __name__ == '__main__':
    # 源文件夹目录
    ori_files_path = '/mnt/d/exp/dji_car/dji_data/'
    ori_beachhaead_1 = ori_files_path + 'Beachhead-1/'
    ori_beachhaead_2 = ori_files_path + 'Beachhead-2/'
    ori_G6 = ori_files_path + 'G6_highway/'
    ori_huailaiexp = ori_files_path + 'Huailai_Experimental_Station/'

    # 源文件夹的标签目录
    ori_label_json = ori_files_path + 'label_json/'
    ori_label_beachhaead_1 = ori_label_json + 'Beachhead-1_label_json/'
    ori_label_beachhaead_2 = ori_label_json + 'Beachhead-2_label_json/'
    ori_label_G6 = ori_label_json + 'G6_highway_label_json/'
    ori_label_huailaiexp = ori_label_json + 'Huailai_Experimental_Station_label_json/'

    # 删除没有目标的图片
    deleteEmptyImg(ori_beachhaead_1, ori_label_beachhaead_1)
    deleteEmptyImg(ori_beachhaead_2, ori_label_beachhaead_2)
    deleteEmptyImg(ori_G6, ori_label_G6)
    deleteEmptyImg(ori_huailaiexp, ori_label_huailaiexp)
    print("删除没有目标的图片完成")

    # 将标签与图片文件放在一个文件夹下
    mix_path = '/mnt/d/exp/dji_car/dji_data_mix/'
    mixFile(ori_files_path, mix_path)
    print("混合图片完成")
    
    # 改变json中的ImagePath属性路径
    changeJsonImagePath(mix_path)
    print("修改ImagePath属性完成")

    # 转换为yolov8所需的txt数据
    label_dict = {"BB": 0, "BS": 1, "DB": 2, "DS": 3}
    datasets_txt = '/mnt/d/exp/dji_car/datasets_label_txt/'
    convertLabelmesToYolo(mix_path, datasets_txt, label_dict)
    print("生成txt格式标注文件完成")

    # 划分为验证集与数据集, 并生成yaml的数据格式
    # Convert label to idx(yolo读取的是idx)
    labeltxt_path = '/mnt/d/exp/dji_car/labels.txt'
    with open(labeltxt_path, "r") as f:
        classes = [c.strip() for c in f.readlines()]
        idx_dict = {c: str(i) for i, c in enumerate(classes)}
    final_datasets = '/mnt/d/exp/dji_car/datasets/'
    split_rate = 0.3    # 验证集占用比例
    splitData(datasets_txt, final_datasets, classes, idx_dict, split_rate)
    print("划分训练集与验证集完成")

    print("全部完成！")

