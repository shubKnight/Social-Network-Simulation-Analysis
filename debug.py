"""Test stability: 10 runs with identical settings to verify no bimodality."""
from engine import SimulationEngine
import numpy as np

STEPS = 500
RUNS = 10

print("=" * 70)
print("STABILITY TEST: 10 runs, T=1.3, p=0, rounds_per_update=5, init=0.8")
print("=" * 70)

rates = []
for i in range(RUNS):
    engine = SimulationEngine(n=100, k=6, p=0.0, T=1.3, R=1.0, P=0.0, S=0.0,
                               init_coop_fraction=0.8, rounds_per_update=5,
                               mutation_rate=0.005)
    for _ in range(STEPS):
        rate = engine.step()
    rates.append(rate)
    print(f"  Run {i+1:2d}: {rate:.0%}")

print(f"\n  Mean: {np.mean(rates):.1%} ± {np.std(rates):.1%}")
print(f"  Min: {min(rates):.0%}, Max: {max(rates):.0%}")

bimodal = any(r < 0.1 for r in rates) and any(r > 0.5 for r in rates)
print(f"\n  Bimodal: {'❌ YES (PROBLEM)' if bimodal else '✅ NO (FIXED!)'}")

print("\n" + "=" * 70)
print("VILLAGE vs CITY: same params")
print("=" * 70)

for label, p_val in [("Village p=0.0", 0.0), ("City p=1.0", 1.0)]:
    run_rates = []
    for _ in range(5):
        engine = SimulationEngine(n=100, k=6, p=p_val, T=1.3, R=1.0, P=0.0, S=0.0,
                                   init_coop_fraction=0.8, rounds_per_update=5,
                                   mutation_rate=0.005)
        for _ in range(STEPS):
            rate = engine.step()
        run_rates.append(rate)
    avg = np.mean(run_rates)
    print(f"  {label}: {avg:.1%} ± {np.std(run_rates):.1%}  ({[f'{r:.0%}' for r in run_rates]})")
