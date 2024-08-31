# Ultralytics YOLO 🚀, AGPL-3.0 license
# ! nn文件目录下的所有文件，是定义模型中的一些组成构建，之后进行改进、优化、增加其他结构时在对应的文件下面进行改动
# !     --autobackend.py 用于自动选择最优的计算后端
# !     --tasks.py 定义了使用神经网络完成的不同任务的流程，如分类、检测或分割。流程定义再次，定义模型前向传播部分

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_scale,
    guess_model_task,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

__all__ = (
    "attempt_load_one_weight",
    "attempt_load_weights",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
)
