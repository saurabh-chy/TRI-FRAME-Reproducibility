# Pilot decision (Adult, 10 runs, mode = full)

ERM clean DP gap: 0.1818
- budget 0.25x: DP under group-targeted attack 0.2097 (+0.0279), full-population 0.2128
- budget 0.5x: DP under group-targeted attack 0.2578 (+0.0760), full-population 0.2681
- budget 1x: DP under group-targeted attack 0.3646 (+0.1828), full-population 0.3896
- budget 2x: DP under group-targeted attack 0.5369 (+0.3551), full-population 0.5836
- budget 4x: DP under group-targeted attack 0.7777 (+0.5959), full-population 0.8527

**Check 1 (attack is meaningful at base budget): YES**

- Fair adv. training (clean + adv. loss): DP under group-targeted attack 0.1476
- Fair adv. training (Chai & Wang 2023): DP under group-targeted attack 0.1607
TRI-FRAME 0.1350 vs strongest baseline (Fair adv. training (clean + adv. loss)) 0.1476; mean paired improvement 0.0126, seed SD 0.0110
Estimated runs needed to detect this effect (paired, alpha 0.05, power 0.8): 6

**Check 2 (TRI-FRAME beats fair adversarial training beyond seed noise): YES**

**Decision:** Method paper: run MODE = 'full' with the complete template.