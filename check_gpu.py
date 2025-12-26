"""Check GPU availability and setup."""
import torch
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

def check_gpu():
    """Check if GPU is available and print info."""
    print("=" * 60)
    print("GPU Setup Check")
    print("=" * 60)
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"\n[OK] GPU Detected!")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Memory info
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1024**3
        print(f"   Total Memory: {total_memory:.2f} GB")
        
        # Test tensor creation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"   [OK] GPU tensor creation: SUCCESS")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   [ERROR] GPU tensor creation: FAILED - {e}")
            return False
        
        print("\n[READY] Ready for GPU training!")
        return True
    else:
        print("\n[WARNING] GPU not available!")
        print("   Reasons could be:")
        print("   1. No NVIDIA GPU installed")
        print("   2. CUDA not installed")
        print("   3. PyTorch not compiled with CUDA support")
        print("\n   To install CUDA-enabled PyTorch:")
        print("   Windows (CUDA 11.8):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   Windows (CUDA 12.1):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

if __name__ == "__main__":
    has_gpu = check_gpu()
    sys.exit(0 if has_gpu else 1)

