from engine import SimulationEngine
import numpy as np
import time

print("=" * 70)
print("GC-MARL MATRIX INTEGRATION TEST")
print("=" * 70)

t0 = time.time()
engine = SimulationEngine(n=100, k=6, p=0.0, T=1.3, R=1.0, P=0.0, S=0.0, 
                          temperature=1.0, temp_decay=0.95)

print(f"Engine Intialized. A_hat env shape: {engine.env_A_hat.shape}, batch shape: {engine.batch_A_hat.shape}")
print(f"Starting training phase...")

# Burn-in phase
for i in range(150):
    rate = engine.step()
    if i % 25 == 0:
        print(f"  Step {i:3d}: Coop Rate={rate:5.1%}, Loss={engine.last_loss:.4f}, Temp={engine.temp:.3f}")

print(f"\nTime to run 150 steps: {time.time() - t0:.2f} seconds")
print("\nGraph Convolution Integration successful!")
