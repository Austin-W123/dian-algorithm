# Level 0 - 环境配置

## 环境

- Windows + WSL2
- Ubuntu 24.04.5 LTS
- Conda 环境：dian-ai
- Python：3.11.9
- PyTorch：2.13.0+cu130
- CUDA Runtime：13.0
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU

## GPU 测试结果

- torch.cuda.is_available(): True
- CUDA Tensor 设备：cuda:0
- GPU 矩阵乘法测试：成功

## 环境记录文件

- environment.yml：Conda 环境记录
- requirements.txt：Python 包记录
