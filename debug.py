"""Debug: verify GC-MARL v3 integration (personalities + reward norm)."""
from engine import SimulationEngine
from collections import Counter
import numpy as np, time

print("=" * 70)
print("GC-MARL v3 — Personality + Reward-Norm Integration Test")
print("=" * 70)

t0 = time.time()
engine = SimulationEngine(n=100, k=6, p=0.0, T=1.3,
                          temperature=2.0, temp_decay=0.99)

print(f"Initialized. A_hat: {engine.env_A_hat.shape}")
print(f"Personality counts: {engine.get_personality_counts()}")
print()
print(f"{'Step':>5}  {'CoopRate':>9}  {'Loss':>10}  {'Temp':>6}  {'Altruist':>9}  {'Grudger':>9}  {'Opportunist':>11}  {'Random':>7}")
print("-" * 75)

for i in range(300):
    rate = engine.step()
    if i % 30 == 0:
        pr = engine.get_personality_coop_rates()
        print(f"{i:>5}  {rate:>9.1%}  {engine.last_loss:>10.4f}  {engine.temp:>6.3f}"
              f"  {pr['altruist']:>9.1%}  {pr['grudger']:>9.1%}"
              f"  {pr['opportunist']:>11.1%}  {pr['random']:>7.1%}")

print()
print(f"Done in {time.time()-t0:.1f}s")

# Check we're not locked at 100%
final_rate = engine.env.get_cooperation_rate()
if 0.1 < final_rate < 0.95:
    print(f"✅ Cooperation rate is {final_rate:.0%} — volatile, realistic!")
elif final_rate >= 0.95:
    print(f"⚠️  Still high ({final_rate:.0%}) — try more steps or higher T")
else:
    print(f"⚠️  Very low ({final_rate:.0%}) — defectors dominating")
