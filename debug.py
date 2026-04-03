"""Sweep p from 0 to 1 to find the phase transition point."""
from engine import SimulationEngine

STEPS = 300
RUNS_PER_P = 3  # Average over multiple runs for stability

print("=" * 60)
print("PHASE TRANSITION SWEEP: p = 0.0 to 1.0")
print("=" * 60)

for p_val in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]:
    avg_rates = []
    for run in range(RUNS_PER_P):
        engine = SimulationEngine(n=100, k=6, p=p_val, T=1.4, R=1.0, P=0.1, S=0.0,
                                   epsilon=0.02, init_defector_fraction=0.1)
        for _ in range(STEPS):
            rate, _ = engine.step()
        avg_rates.append(rate)
    
    mean_rate = sum(avg_rates) / len(avg_rates)
    print(f"  p={p_val:.2f}  →  Final Coop Rate: {mean_rate:.1%}  (runs: {[f'{r:.0%}' for r in avg_rates]})")
