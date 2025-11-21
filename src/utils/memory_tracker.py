"""
Memory tracking utilities for monitoring VRAM usage during training
"""

import torch
import psutil
import os


class MemoryTracker:
    """Track GPU and system memory usage"""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.has_cuda = torch.cuda.is_available()
        
        if self.has_cuda:
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            print(f"CUDA available: {self.device_count} GPU(s)")
            for i, name in enumerate(self.device_names):
                print(f"  GPU {i}: {name}")
        else:
            print("CUDA not available - using CPU")
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        stats = {
            "has_cuda": self.has_cuda,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
        }
        
        if self.has_cuda:
            # Get GPU memory
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            
            # Update peak
            if allocated > self.peak_memory:
                self.peak_memory = allocated
            
            stats.update({
                "allocated_gb": allocated,
                "reserved_gb": reserved,
            })
            
            # Try to get total memory
            try:
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                stats["total_gb"] = total
                stats["free_gb"] = total - allocated
            except Exception:
                pass
        
        # Add system memory
        system_memory = psutil.virtual_memory()
        stats["system_memory_gb"] = system_memory.total / (1024 ** 3)
        stats["system_memory_used_gb"] = system_memory.used / (1024 ** 3)
        stats["system_memory_percent"] = system_memory.percent
        
        return stats
    
    def get_peak_memory(self):
        """Get peak memory usage"""
        if self.has_cuda:
            # Update peak before returning
            current = torch.cuda.memory_allocated() / (1024 ** 3)
            if current > self.peak_memory:
                self.peak_memory = current
            return self.peak_memory
        return 0.0
    
    def reset_peak_memory(self):
        """Reset peak memory counter"""
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0.0
    
    def print_memory_stats(self):
        """Print formatted memory statistics"""
        stats = self.get_memory_stats()
        
        print("\n" + "="*50)
        print("Memory Statistics")
        print("="*50)
        
        if stats["has_cuda"]:
            print(f"GPU Memory:")
            print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
            if stats['total_gb'] > 0:
                print(f"  Total:     {stats['total_gb']:.2f} GB")
                print(f"  Free:      {stats['free_gb']:.2f} GB")
            print(f"  Peak:      {self.peak_memory:.2f} GB")
        else:
            print("GPU: Not available")
        
        print(f"\nSystem Memory:")
        print(f"  Total: {stats['system_memory_gb']:.2f} GB")
        print(f"  Used:  {stats['system_memory_used_gb']:.2f} GB ({stats['system_memory_percent']:.1f}%)")
        print("="*50 + "\n")
