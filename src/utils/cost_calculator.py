"""
Cost calculation utilities for comparing training costs
"""


class CostCalculator:
    """Calculate training costs based on GPU hours and memory usage"""
    
    # GPU pricing per hour (approximate AWS/GCP costs)
    GPU_COSTS = {
        "A100_40GB": 4.00,   # per hour
        "A100_80GB": 5.00,   # per hour
        "V100": 2.50,        # per hour
        "T4": 0.50,          # per hour
        "A10": 1.00,         # per hour
    }
    
    def __init__(self, gpu_type="A100_40GB"):
        """
        Initialize cost calculator
        
        Args:
            gpu_type: Type of GPU being used
        """
        self.gpu_type = gpu_type
        self.hourly_rate = self.GPU_COSTS.get(gpu_type, self.GPU_COSTS["A100_40GB"])
    
    def calculate_training_cost(self, time_seconds, peak_memory_gb, method):
        """
        Calculate training cost
        
        Args:
            time_seconds: Training time in seconds
            peak_memory_gb: Peak memory usage in GB
            method: Training method (qlora, spectrum, etc.)
        
        Returns:
            Cost in USD
        """
        # Convert to hours
        hours = time_seconds / 3600
        
        # Base cost
        cost = hours * self.hourly_rate
        
        # Add efficiency factor based on method
        efficiency_factors = {
            "qlora": 0.8,      # 20% more efficient
            "spectrum": 0.75,  # 25% more efficient
            "full": 1.0,       # baseline
        }
        
        efficiency = efficiency_factors.get(method, 1.0)
        cost *= efficiency
        
        return cost
    
    def calculate_cost_per_epoch(self, total_cost, num_epochs):
        """Calculate cost per epoch"""
        return total_cost / num_epochs if num_epochs > 0 else 0.0
    
    def calculate_savings(self, baseline_cost, optimized_cost):
        """
        Calculate cost savings
        
        Returns:
            Dictionary with absolute and percentage savings
        """
        savings = baseline_cost - optimized_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        
        return {
            "absolute_savings": savings,
            "percent_savings": savings_percent,
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
        }
    
    def estimate_total_cost(self, time_per_epoch_seconds, num_epochs, memory_gb, method):
        """
        Estimate total training cost
        
        Args:
            time_per_epoch_seconds: Time per epoch in seconds
            num_epochs: Number of epochs
            memory_gb: Memory usage in GB
            method: Training method
        
        Returns:
            Dictionary with cost breakdown
        """
        cost_per_epoch = self.calculate_training_cost(time_per_epoch_seconds, memory_gb, method)
        total_cost = cost_per_epoch * num_epochs
        
        return {
            "cost_per_epoch": cost_per_epoch,
            "total_cost": total_cost,
            "num_epochs": num_epochs,
            "hourly_rate": self.hourly_rate,
            "gpu_type": self.gpu_type,
            "method": method,
        }
    
    def compare_methods(self, results_dict):
        """
        Compare costs across multiple training methods
        
        Args:
            results_dict: Dictionary mapping method names to their metrics
            
        Returns:
            Comparison statistics
        """
        comparisons = {}
        
        # Find baseline (full finetuning or first method)
        baseline_method = "full" if "full" in results_dict else list(results_dict.keys())[0]
        baseline_cost = results_dict[baseline_method].get("total_cost", 0)
        
        for method, metrics in results_dict.items():
            cost = metrics.get("total_cost", 0)
            savings = self.calculate_savings(baseline_cost, cost)
            
            comparisons[method] = {
                "cost": cost,
                "savings_vs_baseline": savings["absolute_savings"],
                "savings_percent": savings["percent_savings"],
                "memory_gb": metrics.get("peak_memory_gb", 0),
                "training_time": metrics.get("total_training_time", 0),
            }
        
        return {
            "baseline_method": baseline_method,
            "comparisons": comparisons,
        }
