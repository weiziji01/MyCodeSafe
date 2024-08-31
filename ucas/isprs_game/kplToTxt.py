# mmrotate中test输出的kpl文件转换为txt标签，为dota格式标注
import os
import math
import mmcv
 
result = mmcv.load('/mnt/d/isprs_game/result/try_c4/test_easy/ann.pkl')
class_map = ['ship','vehicle','airplane','other']
draw = r'/mnt/d/isprs_game/result/try_c4/test_easy/image'
labelTxt = r'/mnt/d/isprs_game/result/try_c4/test_easy/annTxt'
score_thr = 0.3
def rota(x, y, w, h, a):  # 旋转中心点，旋转中心点，框的w，h，旋转角
    center_x1 = x
    center_y1 = y
    x1, y1 = x - w / 2, y - h / 2  # 旋转前左上
    x2, y2 = x + w / 2, y - h / 2  # 旋转前右上
    x3, y3 = x + w / 2, y + h / 2  # 旋转前右下
    x4, y4 = x - w / 2, y + h / 2  # 旋转前左下
    px1 = (x1 - center_x1) * math.cos(a) - (y1 - center_y1) * math.sin(a) + center_x1  # 旋转后左上
    py1 = (x1 - center_x1) * math.sin(a) + (y1 - center_y1) * math.cos(a) + center_y1
    px2 = (x2 - center_x1) * math.cos(a) - (y2 - center_y1) * math.sin(a) + center_x1  # 旋转后右上
    py2 = (x2 - center_x1) * math.sin(a) + (y2 - center_y1) * math.cos(a) + center_y1
    px3 = (x3 - center_x1) * math.cos(a) - (y3 - center_y1) * math.sin(a) + center_x1  # 旋转后右下
    py3 = (x3 - center_x1) * math.sin(a) + (y3 - center_y1) * math.cos(a) + center_y1
    px4 = (x4 - center_x1) * math.cos(a) - (y4 - center_y1) * math.sin(a) + center_x1  # 旋转后左下
    py4 = (x4 - center_x1) * math.sin(a) + (y4 - center_y1) * math.cos(a) + center_y1
 
    return px1, py1, px2, py2, px3, py3, px4, py4  # 旋转后的四个点,左上，右上，右下，左下
 
 
def mmrotate2dota(result, class_map, score_thr,img_path,txt_path):
    for i,info in enumerate(result):
        filename = os.listdir(img_path)[i][:-4]+".txt"
        out_path= os.path.join(txt_path,filename)
        info_list = []
        for cla,val in enumerate(info):
            for j in range(len(val)):
                x = float(val[j][0])
                y = float(val[j][1])
                w = float(val[j][2])
                h = float(val[j][3])
                a = float(val[j][4])
                score = float(val[j][5])
                px1, py1, px2, py2, px3, py3, px4, py4 = rota(x, y, w, h, a)
                px1 = round(px1)/1.0
                py1 = round(py1)/1.0
                px2 = round(px2)/1.0
                py2 = round(py2)/1.0
                px3 = round(px3)/1.0
                py3 = round(py3)/1.0
                px4 = round(px4)/1.0
                py4 = round(py4)/1.0
 
                # 目标格式为  x1、y1、x2、y2、x3、y3、x4、y4、 classname、diffcult
                dstline = str(px1) + " " + str(py1) + " " + str(px2) + " " + str(py2) + " " + str(px3) + " " + str(
                    py3) + " " + str(px4) + " " + str(py4) + " " + str(class_map[cla]) + " " + "0"
                if(score >= score_thr):
                    info_list.append(dstline)
        with open(out_path, 'w') as fw:
            fw.writelines([line+'\n' for line in info_list]) #添加换行
 
 
 
mmrotate2dota(result, class_map,score_thr,draw,labelTxt)
print('convertdone')