import torch
print("CUDA 可用：", torch.cuda.is_available())
print("GPU 名称：", torch.cuda.get_device_name(0))
print("PyTorch CUDA 版本：", torch.version.cuda)