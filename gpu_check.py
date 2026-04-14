import torch

# 1. 检查 CUDA 是否驱动正常
cuda_available = torch.cuda.is_available()
print(f"GPU 是否可用: {cuda_available}")

if cuda_available:
    # 2. 获取 GPU 设备名称
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    # 3. 简单的矩阵运算测试 (真正确认能跑通)
    x = torch.rand(5, 3).to("cuda")
    print("张量已成功移动到 GPU:")
    print(x)
else:
    print("目前只能使用 CPU。请检查驱动或 PyTorch 安装版本。")