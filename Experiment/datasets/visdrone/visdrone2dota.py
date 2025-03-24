"""
VisDrone标注转为DOTA标注格式
"""
import os
from PIL import ImageDraw, Image


VisDrone_Classes = [
    'ignored-regions',
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
    'others'
]
nums_dict = {}
for e in VisDrone_Classes:
    nums_dict[e] = 0

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def draw_bboxes(img, bboxes):
    draw = ImageDraw.Draw(img)

    for box in bboxes:
        x, y, w, h = box
        # draw.rectangle((
        #     (x - w / 2, y - h / 2),
        #     (x + w / 2, y + h / 2)), fill=None, outline=(255, 0, 0), width=1)
        draw.rectangle((
            (x, y),
            (x + w, y + h)), fill=None, outline=(255, 0, 0), width=1)
    img.show()



# old_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/train/annotations'
# img_path = '/mnt/d/data/visdrone2019/visdrone2019/train/images'
# new_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/train/labels'


# old_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/test/annotations'
# img_path = '/mnt/d/data/visdrone2019/visdrone2019/test/images'
# new_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/test/labels'


old_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/val/annotations'
img_path = '/mnt/d/data/visdrone2019/visdrone2019/val/images'
new_txt_path = '/mnt/d/data/visdrone2019/visdrone2019/val/labels'

check_and_mkdir(new_txt_path)

for e in os.listdir(old_txt_path):
    txt_file = os.path.join(old_txt_path, e)
    new_txt_file = os.path.join(new_txt_path, e)
    out = []
    temp_bbox = []
    with open(txt_file, 'r') as f:
        for l in f.readlines():
            ele = l.split(',')
            x = float(ele[0])
            y = float(ele[1])
            w = float(ele[2])
            h = float(ele[3])
            cat_id = int(ele[5])
            xmin = str(x)
            xmax = str(x + w)
            ymin = str(y)
            ymax = str(y + h)
            poly = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            poly.append(cat_id)

            nums_dict[VisDrone_Classes[cat_id]] += 1

            temp_bbox.append([x, y, w, h])
            out.append(poly)
    
    with open(new_txt_file, 'w') as f:
        for e in out:
            if e[-1] not in [0, 11]:
                info = ' '.join(e[:-1]) + ' ' + VisDrone_Classes[e[-1]] + ' 0' + '\n'
                f.write(info)

print(nums_dict)