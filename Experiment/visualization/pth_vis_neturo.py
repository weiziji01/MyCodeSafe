#
# ! 使用netron工具可视化pytorch模型
import netron
import torch

def pth2onnx(pth_path, onnx_path):
    dummy_input = torch.randn(1, 3, 224,224, device='cuda')
    model = torch.load(pth_path, map_location='cuda:0')
    input_model_names = ['input']
    output_model_names = ['output']

    torch.onnx.export(model, dummy_input, onnx_path,verbose=True, 
                      input_names=input_model_names, output_names=output_model_names)
    return 0

if __name__ == '__main__':
    pytorch_model_path = "/mnt/d/exp/sodaa_sob/a6000result/0907/epoch_9.pth"
    onnx_model_path = "/mnt/d/exp/sodaa_sob/a6000result/0907/epoch_9.onnx"
    pth2onnx(pytorch_model_path, onnx_model_path)
    netron.start(onnx_model_path)