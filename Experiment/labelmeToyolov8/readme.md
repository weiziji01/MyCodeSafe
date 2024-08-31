**顺序：**

1.Tojpg.py -- 将图片格式改为jpg（原图是JPG，不改应该也可以）

2.deleteEmptyImg.py -- 删除没有标签的图片

3.mixFile.py -- 将图片和标签放在同一个文件夹中(如mix)

4.changeJsonImagePath.py -- 修改labelme生成的json标签中的图片路径ImagePath那一项，会改到当前的文件夹中

5.创建一个label.txt文件，里面有类别和序号信息(labelme2yolov8txt.py文件需要)

5.labelme2yolov8txt.py -- 将mix文件夹的labelme标签转成yolov8格式。会生成两个文件夹，一个是全都是转换为txt标注的文件夹(output_folder)，一个是划分成train和val并有yaml格式的文件夹(dataset_folder).