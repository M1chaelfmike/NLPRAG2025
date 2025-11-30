import subprocess
import sys
import torch
import platform
import re
from typing import Tuple, Optional, Dict

# 全局配置：GPU型号→计算能力→推荐CUDA版本映射（覆盖主流NVIDIA显卡）
# 计算能力参考：https://developer.nvidia.com/cuda-gpus
NVIDIA_GPU_CONFIG: Dict[str, Dict[str, str]] = {
    # RTX 40系列
    "RTX 4090": {"compute_capability": "8.9", "min_cuda": "11.8", "recommend_cuda": "12.1"},
    "RTX 4080": {"compute_capability": "8.9", "min_cuda": "11.8", "recommend_cuda": "12.1"},
    "RTX 4070": {"compute_capability": "8.9", "min_cuda": "11.8", "recommend_cuda": "12.1"},
    "RTX 4060": {"compute_capability": "8.9", "min_cuda": "11.8", "recommend_cuda": "12.1"},
    "RTX 4050": {"compute_capability": "8.9", "min_cuda": "11.8", "recommend_cuda": "12.1"},
    # RTX 30系列
    "RTX 3090": {"compute_capability": "8.6", "min_cuda": "11.1", "recommend_cuda": "11.8"},
    "RTX 3080": {"compute_capability": "8.6", "min_cuda": "11.1", "recommend_cuda": "11.8"},
    "RTX 3070": {"compute_capability": "8.6", "min_cuda": "11.1", "recommend_cuda": "11.8"},
    "RTX 3060": {"compute_capability": "8.6", "min_cuda": "11.1", "recommend_cuda": "11.8"},
    "RTX 3050": {"compute_capability": "8.6", "min_cuda": "11.1", "recommend_cuda": "11.8"},
    # RTX 20系列
    "RTX 2080": {"compute_capability": "7.5", "min_cuda": "10.2", "recommend_cuda": "11.7"},
    "RTX 2070": {"compute_capability": "7.5", "min_cuda": "10.2", "recommend_cuda": "11.7"},
    "RTX 2060": {"compute_capability": "7.5", "min_cuda": "10.2", "recommend_cuda": "11.7"},
    # GTX系列
    "GTX 1660": {"compute_capability": "7.5", "min_cuda": "10.2", "recommend_cuda": "11.3"},
    "GTX 1080": {"compute_capability": "6.1", "min_cuda": "8.0", "recommend_cuda": "11.1"},
    "GTX 1070": {"compute_capability": "6.1", "min_cuda": "8.0", "recommend_cuda": "11.1"},
    "GTX 1060": {"compute_capability": "6.1", "min_cuda": "8.0", "recommend_cuda": "11.1"},
    # 其他常见型号
    "Tesla V100": {"compute_capability": "7.0", "min_cuda": "9.0", "recommend_cuda": "11.3"},
    "Tesla T4": {"compute_capability": "7.5", "min_cuda": "10.2", "recommend_cuda": "11.7"},
}

# PyTorch与CUDA/ROCm版本映射（确保兼容性）
PYTORCH_VERSION_MAP: Dict[str, Dict[str, str]] = {
    "cuda": {
        "11.1": "torch==1.9.1+cu111",
        "11.3": "torch==1.12.1+cu113",
        "11.7": "torch==1.13.1+cu117",
        "11.8": "torch==2.0.1+cu118",
        "12.1": "torch==2.1.0+cu121",
    },
    "rocm": {
        "5.6": "torch==2.0.1+rocm5.6",
        "5.7": "torch==2.1.0+rocm5.7",
    },
}

# ROCm支持的AMD显卡（参考：https://docs.amd.com/en/docs-versions/rocm-5.7.0/reference/gpu-accelerated.html）
AMD_ROCM_GPUS = [
    "Radeon RX 6000系列", "Radeon RX 7000系列", "Radeon Pro V620",
    "Instinct MI50", "Instinct MI60", "Instinct MI250"
]


class UniversalGPUAcceleratorChecker:
    def __init__(self):
        self.os_type = platform.system()  # Windows/Linux/Darwin(Mac)
        self.gpu_vendor = self._detect_gpu_vendor()  # NVIDIA/AMD/Unknown
        self.gpu_model = self._detect_gpu_model()  # 具体GPU型号

    def _detect_gpu_vendor(self) -> str:
        """检测GPU厂商（NVIDIA/AMD/Unknown）"""
        try:
            if torch.cuda.is_available():
                return "NVIDIA"
            # 检测AMD显卡（Windows/Linux）
            if self.os_type == "Windows":
                result = subprocess.check_output(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    text=True, stderr=subprocess.STDOUT
                )
                if any("AMD" in line or "Radeon" in line for line in result.splitlines()):
                    return "AMD"
            elif self.os_type == "Linux":
                result = subprocess.check_output(
                    ["lspci"], text=True, stderr=subprocess.STDOUT, shell=True
                )
                if any("AMD" in line or "Radeon" in line for line in result.splitlines()):
                    return "AMD"
            return "Unknown"
        except Exception:
            return "Unknown"

    def _detect_gpu_model(self) -> Optional[str]:
        """检测具体GPU型号（如RTX 3060、Radeon RX 6800 XT）"""
        try:
            if self.gpu_vendor == "NVIDIA":
                if torch.cuda.is_available():
                    return torch.cuda.get_device_name(0)
                # 备选：通过nvidia-smi检测
                result = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
                for line in result.splitlines():
                    if "NVIDIA GeForce" in line or "NVIDIA RTX" in line:
                        return line.strip().split(" ")[-1]  # 提取型号
            elif self.gpu_vendor == "AMD":
                if self.os_type == "Windows":
                    result = subprocess.check_output(
                        ["wmic", "path", "win32_videocontroller", "get", "name"],
                        text=True, stderr=subprocess.STDOUT
                    )
                    for line in result.splitlines():
                        if "AMD" in line or "Radeon" in line:
                            return line.strip()
                elif self.os_type == "Linux":
                    result = subprocess.check_output(
                        ["lspci | grep -i vga"], text=True, stderr=subprocess.STDOUT, shell=True
                    )
                    return result.strip()
            return None
        except Exception:
            return None

    def _get_nvidia_recommended_cuda(self) -> str:
        """根据NVIDIA GPU型号推荐最佳CUDA版本"""
        if not self.gpu_model:
            return "11.8"  # 默认推荐稳定版
        # 模糊匹配GPU型号（如"RTX 3060 Laptop"匹配"RTX 3060"）
        for model_pattern, config in NVIDIA_GPU_CONFIG.items():
            if model_pattern in self.gpu_model:
                return config["recommend_cuda"]
        # 未匹配到具体型号，推荐兼容性最广的11.8
        return "11.8"

    def _get_amd_recommended_rocm(self) -> str:
        """根据AMD GPU推荐最佳ROCm版本"""
        return "5.6"  # 兼容大部分AMD显卡的稳定版

    def check_cuda_available(self) -> Tuple[bool, str]:
        """检查CUDA/ROCm是否可用及版本信息"""
        try:
            if self.gpu_vendor == "NVIDIA":
                # 检查NVIDIA CUDA
                torch_cuda_available = torch.cuda.is_available()
                torch_cuda_version = torch.version.cuda if torch_cuda_available else None

                # 检查系统层面CUDA
                system_cuda_available = False
                system_cuda_version = None
                try:
                    cmd = ["nvcc", "--version"]
                    if self.os_type != "Windows":
                        cmd.append("--shell")
                    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    version_match = re.search(r"release (\d+\.\d+)", result)
                    if version_match:
                        system_cuda_version = version_match.group(1)
                        system_cuda_available = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

                # 综合判断
                if torch_cuda_available and system_cuda_available:
                    return True, f"PyTorch CUDA {torch_cuda_version} (系统CUDA {system_cuda_version})"
                elif system_cuda_available and not torch_cuda_available:
                    return False, f"系统已安装CUDA {system_cuda_version}，但PyTorch未使用GPU版本"
                elif not system_cuda_available:
                    return False, "未检测到系统CUDA环境"
                else:
                    return False, "未知错误"

            elif self.gpu_vendor == "AMD":
                # 检查AMD ROCm
                torch_rocm_available = torch.backends.rocm.is_available()
                torch_rocm_version = torch.version.rocm if torch_rocm_available else None
                if torch_rocm_available:
                    return True, f"PyTorch ROCm {torch_rocm_version}"
                else:
                    return False, "未检测到ROCm环境（AMD显卡需安装ROCm替代CUDA）"

            else:
                # 无GPU或未知显卡
                return False, "未检测到支持GPU加速的显卡（仅NVIDIA/AMD显卡支持）"
        except Exception as e:
            return False, f"检测失败: {str(e)}"

    def get_gpu_info(self) -> str:
        """获取GPU详细信息（型号、显存、厂商）"""
        try:
            if self.gpu_vendor == "NVIDIA":
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    return (
                        f"厂商: NVIDIA\n"
                        f"型号: {props.name}\n"
                        f"显存: {props.total_memory / 1024 ** 3:.2f}GB\n"
                        f"计算能力: {props.major}.{props.minor}\n"
                        f"推荐CUDA版本: {self._get_nvidia_recommended_cuda()}"
                    )
                else:
                    result = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
                    gpu_line = [line for line in result.splitlines() if "NVIDIA" in line][0]
                    return f"厂商: NVIDIA\n型号: {gpu_line.strip()}\n推荐CUDA版本: {self._get_nvidia_recommended_cuda()}"

            elif self.gpu_vendor == "AMD":
                gpu_model = self.gpu_model or "未知AMD显卡"
                return (
                    f"厂商: AMD\n"
                    f"型号: {gpu_model}\n"
                    f"推荐ROCm版本: {self._get_amd_recommended_rocm()}\n"
                    f"提示: AMD显卡需安装ROCm替代CUDA"
                )

            else:
                return (
                    f"厂商: 未知\n"
                    f"型号: 未检测到支持GPU加速的显卡\n"
                    f"提示: 仅NVIDIA/AMD显卡支持GPU加速，当前仅能使用CPU"
                )
        except Exception as e:
            return f"获取GPU信息失败: {str(e)}"

    def install_instructions(self) -> str:
        """生成适配当前硬件的安装指导"""
        instructions = []
        if self.gpu_vendor == "NVIDIA":
            cuda_version = self._get_nvidia_recommended_cuda()
            instructions.append(f"=== 推荐安装CUDA {cuda_version}（适配你的{self.gpu_model}）===")

            if self.os_type == "Windows":
                instructions.append(f"1. 下载CUDA {cuda_version}安装包:")
                instructions.append(f"   https://developer.nvidia.com/cuda-{cuda_version}-0-download-archive")
                instructions.append("2. 安装时勾选:")
                instructions.append("   - CUDA Runtime (必需)")
                instructions.append("   - cuDNN (可选但推荐，加速模型推理)")
                instructions.append("   - 笔记本用户务必勾选'笔记本优化'")
                instructions.append("3. 验证安装:")
                instructions.append("   打开命令提示符，输入: nvcc --version")

            elif self.os_type == "Linux":
                instructions.append(f"1. 运行自动安装脚本（Ubuntu/Debian）:")
                instructions.append(
                    f"   wget https://developer.download.nvidia.com/compute/cuda/{cuda_version}.0/local_installers/cuda_{cuda_version}.0_525.85.12_linux.run")
                instructions.append(f"   sudo sh cuda_{cuda_version}.0_525.85.12_linux.run")
                instructions.append("2. 配置环境变量（添加到~/.bashrc或~/.zshrc）:")
                instructions.append(f"   echo 'export PATH=/usr/local/cuda-{cuda_version}/bin:$PATH' >> ~/.bashrc")
                instructions.append(
                    f"   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-{cuda_version}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc")
                instructions.append("   source ~/.bashrc")

            # PyTorch安装指令
            torch_cmd = PYTORCH_VERSION_MAP["cuda"].get(cuda_version, PYTORCH_VERSION_MAP["cuda"]["11.8"])
            instructions.append(f"\n=== 安装GPU版PyTorch（适配CUDA {cuda_version}）===")
            instructions.append(
                f"pip install {torch_cmd} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}")

        elif self.gpu_vendor == "AMD":
            rocm_version = self._get_amd_recommended_rocm()
            instructions.append(f"=== 推荐安装ROCm {rocm_version}（适配你的AMD显卡）===")

            if self.os_type == "Linux":
                # AMD ROCm仅支持Linux
                instructions.append("1. 安装ROCm依赖（Ubuntu 20.04/22.04）:")
                instructions.append("   sudo apt update && sudo apt install wget gnupg2")
                instructions.append(f"   wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -")
                instructions.append(
                    f"   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version} focal main' | sudo tee /etc/apt/sources.list.d/rocm.list")
                instructions.append("   sudo apt update && sudo apt install rocm-hip-sdk rocm-opencl-sdk")
                instructions.append("2. 配置环境变量:")
                instructions.append("   echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc")
                instructions.append("   echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc")
                instructions.append("   source ~/.bashrc")

                # PyTorch安装指令
                torch_cmd = PYTORCH_VERSION_MAP["rocm"][rocm_version]
                instructions.append(f"\n=== 安装GPU版PyTorch（适配ROCm {rocm_version}）===")
                instructions.append(
                    f"pip install {torch_cmd} torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm{rocm_version.replace('.', '')}")
            else:
                instructions.append("⚠️  注意：AMD ROCm仅支持Linux系统，Windows/MacOS暂不支持GPU加速")

        else:
            instructions.append("=== 无可用GPU加速方案 ===")
            instructions.append("当前设备未检测到支持GPU加速的显卡（仅NVIDIA/AMD显卡支持）")
            instructions.append("建议：使用CPU模式运行，或更换NVIDIA/AMD显卡")

        return "\n".join(instructions)

    def verify_installation(self) -> bool:
        """验证GPU加速是否配置成功"""
        if self.gpu_vendor == "NVIDIA":
            return torch.cuda.is_available()
        elif self.gpu_vendor == "AMD":
            return torch.backends.rocm.is_available()
        else:
            return False

    def run_full_check(self) -> None:
        """运行完整检测+安装指导流程"""
        print("=" * 50)
        print("🎯 通用GPU加速环境检测工具（支持NVIDIA/AMD全系列显卡）")
        print("=" * 50)

        # 1. 显示GPU基础信息
        print("\n📊 GPU硬件信息:")
        print("-" * 30)
        print(self.get_gpu_info())

        # 2. 检查加速环境状态
        print("\n🔍 加速环境检测结果:")
        print("-" * 30)
        cuda_available, cuda_msg = self.check_cuda_available()
        print(f"加速状态: {'✅ 可用' if cuda_available else '❌ 不可用'}")
        print(f"详细信息: {cuda_msg}")

        # 3. 生成安装指导（仅当加速不可用时）
        if not cuda_available and self.gpu_vendor in ["NVIDIA", "AMD"]:
            print("\n📋 适配安装指导:")
            print("-" * 30)
            print(self.install_instructions())

            # 4. 询问是否验证安装（Linux系统支持直接执行命令）
            if self.os_type == "Linux":
                confirm = input("\n是否要在终端中显示完整安装命令？(y/n): ").strip().lower()
                if confirm == "y":
                    print("\n💻 完整安装命令:")
                    print("-" * 30)
                    if self.gpu_vendor == "NVIDIA":
                        cuda_version = self._get_nvidia_recommended_cuda()
                        print(f"# 安装CUDA {cuda_version}")
                        print(
                            f"wget https://developer.download.nvidia.com/compute/cuda/{cuda_version}.0/local_installers/cuda_{cuda_version}.0_525.85.12_linux.run")
                        print(f"sudo sh cuda_{cuda_version}.0_525.85.12_linux.run")
                        print(f"echo 'export PATH=/usr/local/cuda-{cuda_version}/bin:$PATH' >> ~/.bashrc")
                        print(
                            f"echo 'export LD_LIBRARY_PATH=/usr/local/cuda-{cuda_version}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc")
                        print("source ~/.bashrc")
                        # PyTorch命令
                        torch_cmd = PYTORCH_VERSION_MAP["cuda"][cuda_version]
                        print(f"\n# 安装PyTorch（适配CUDA {cuda_version}）")
                        print(
                            f"pip install {torch_cmd} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}")
                    elif self.gpu_vendor == "AMD":
                        rocm_version = self._get_amd_recommended_rocm()
                        print("# 安装ROCm 5.6")
                        print("sudo apt update && sudo apt install wget gnupg2")
                        print(f"wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -")
                        print(
                            f"echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version} focal main' | sudo tee /etc/apt/sources.list.d/rocm.list")
                        print("sudo apt update && sudo apt install rocm-hip-sdk rocm-opencl-sdk")
                        print("echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc")
                        print("echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc")
                        print("source ~/.bashrc")
                        # PyTorch命令
                        torch_cmd = PYTORCH_VERSION_MAP["rocm"][rocm_version]
                        print(f"\n# 安装PyTorch（适配ROCm {rocm_version}）")
                        print(
                            f"pip install {torch_cmd} torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm{rocm_version.replace('.', '')}")

        # 5. 最终验证结果
        print("\n" + "=" * 50)
        print("✅ 检测完成！")
        if self.verify_installation():
            print("🎉 GPU加速环境已就绪，可正常用于idense检索等模块！")
        else:
            print("⚠️  请按照上述指导完成安装，安装后重新运行本工具验证配置。")
        print("=" * 50)


if __name__ == "__main__":
    checker = UniversalGPUAcceleratorChecker()
    checker.run_full_check()