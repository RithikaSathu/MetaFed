import torch
import psutil
import GPUtil

class GPUManager:
    def __init__(self):
        self.device = self._setup_device()
        self._print_gpu_info()
    
    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Set optimal GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Memory management
            torch.cuda.empty_cache()
            
            return device
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _print_gpu_info(self):
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Memory info
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU Memory - Total: {gpu.memoryTotal}MB, Free: {gpu.memoryFree}MB, Used: {gpu.memoryUsed}MB")
        else:
            print("No GPU available, using CPU")
    
    def get_device(self):
        return self.device
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        return None