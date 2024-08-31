
# * 怀来无人机照片训练的测试

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

# 加载模型
model = YOLO("yolov8x-car.yaml")
model = YOLO("yolov8x.pt")
model = YOLO("yolov8x-car.yaml").load("yolov8x.pt")

# 数据集路径
datasets = '/mnt/d/exp/dji_car/datasets/data.yaml'

# 设置输出结果
current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_CFG.save_dir = f'/mnt/d/exp/dji_car/{time_str}'

# 训练模型
results = model.train(data=datasets, epochs=1, imgsz=640,batch=1)

