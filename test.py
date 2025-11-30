import torch
print("CUDA 可用：", torch.cuda.is_available())  # 应输出 True
print("GPU 名称：", torch.cuda.get_device_name(0))  # 应输出 "NVIDIA GeForce RTX 3060"
print("PyTorch CUDA 版本：", torch.version.cuda)  # 应输出 11.8