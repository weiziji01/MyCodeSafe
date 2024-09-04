python car_allcolor.py | tee train.log


**2024-0821_23-08-09: (使用的是datasets_have_error_split，253张训练，76张验证)**
    
    --models: yolov8x, 
    --epochs: 300,
    --imgsz: 640,
    --batch: 1


**2024-08-22_20-11-44: (忘记了添加tensorboard，且修改val.py失败，177张训练，50张验证，26张测试)**

    --models: yolov8x,
    --epochs: 500,
    --imgsz: 3000,
    --batch: 1


