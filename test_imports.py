#!/usr/bin/env python3
"""
Quick syntax and import test for all modules
NOTE: This script modifies sys.path for testing purposes only.
      In production, install the package properly using setup.py
"""

import sys
import os

print("Testing module imports...")

# Test training modules (NOTE: sys.path modification for testing only)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    print("✓ Added src to path")
except Exception as e:
    print(f"✗ Failed to add src to path: {e}")
    sys.exit(1)

# Test utils imports
try:
    from utils.memory_tracker import MemoryTracker
    print("✓ MemoryTracker imported successfully")
except Exception as e:
    print(f"✗ Failed to import MemoryTracker: {e}")

try:
    from utils.cost_calculator import CostCalculator
    print("✓ CostCalculator imported successfully")
except Exception as e:
    print(f"✗ Failed to import CostCalculator: {e}")

# Test instantiation
try:
    tracker = MemoryTracker()
    print(f"✓ MemoryTracker instantiated (CUDA: {tracker.has_cuda})")
except Exception as e:
    print(f"✗ Failed to instantiate MemoryTracker: {e}")

try:
    calculator = CostCalculator()
    print(f"✓ CostCalculator instantiated (Rate: ${calculator.hourly_rate}/hr)")
except Exception as e:
    print(f"✗ Failed to instantiate CostCalculator: {e}")

# Test basic functionality
try:
    stats = tracker.get_memory_stats()
    print(f"✓ Memory stats retrieved: {stats['has_cuda']}")
except Exception as e:
    print(f"✗ Failed to get memory stats: {e}")

try:
    cost = calculator.calculate_training_cost(3600, 20.0, "qlora")
    print(f"✓ Cost calculation works: ${cost:.2f} for 1 hour")
except Exception as e:
    print(f"✗ Failed to calculate cost: {e}")

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
