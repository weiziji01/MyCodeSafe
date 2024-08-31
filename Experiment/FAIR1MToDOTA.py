# fair1m数据集的标注格式改为与dota类似的，即xml改为txt
import os
import sys
import xml.etree.ElementTree as ET
import glob

def xml_to_txt(indir, outdir):

    # os.chdir(indir)      #  切换到给定目录下        
    annotations = os.listdir(indir)

    for i, file in enumerate(annotations):

        file_save = file.split('.')[0]+'.txt'
        file_txt=os.path.join(outdir, file_save)
        f_w = open(file_txt, mode='w+', encoding='UTF-8')

        # actual parsing
        in_file = open(indir + '/' + file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            current = list()
            name = obj.find('possibleresult').find('name').text
            name = name.replace(" ", "-")  #FAIR1M的类别标注会有空格，用-代替
            name_new=update_cate(name)

            xmlbox = obj.find('points')
            points = xmlbox.findall('point')

            poly_list = []
            for point in points:
                point = point.text
                point = point.split(",")
                poly_list = poly_list + point
            f_w.write(poly_list[0] + ' ' + poly_list[1] + ' '+ poly_list[2] + ' ' + poly_list[3] + ' ' +
                       poly_list[4] + ' '+ poly_list[5] + ' ' + poly_list[6] + ' ' + poly_list[7] + ' ' + name_new + ' ' + '0' + '\n')
        f_w.close()

def update_cate(n):
    name_new=' '
    if n == 'Cargo-Truck' or n == 'Small-Car' or n == 'Dump-Truck' or n == 'Van' or n == 'Excavator' or n == 'other-vehicle' or n == 'Bus' or n == 'Tractor' or n == 'Trailer' or n == 'Truck-Tractor':
        name_new = 'vehicle'
    elif n == 'Liquid-Cargo-Ship' or n == 'Passenger-Ship' or n == 'Dry-Cargo-Ship' or n == 'Motorboat' or n == 'other-ship' or n == 'Engineering-Ship' or n == 'Warship'or n == 'Tugboat' or n == 'Fishing-Boat':
        name_new = 'ship'
    elif n == 'Boeing737' or n == 'A321' or n == 'A220' or n == 'other-airplane' or n == 'Boeing787' or n == 'ARJ21' or n == 'Boeing747' or n == 'A330' or n == 'A350' or n == 'Boeing777' or n == 'C919':
        name_new = 'airplane'
    # elif n == 'Bridgr' or n == 'Intersection' or n == 'Tennis-Court' or n == 'Basketball-Court' or n == 'Football-Field' or n == 'Roundabout' or n == 'Baseball-Field':
    else:
        name_new = 'other'
    return name_new

indir='/mnt/d/data/fair1m_exp/train/labelXml'   #xml目录
outdir='/mnt/d/data/fair1m_exp/train/labelTxt'  #txt目录

if __name__ == "__main__":
    xml_to_txt(indir, outdir)
