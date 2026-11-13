# Normalised hypervolume (clean accuracy vs DP under group-targeted attack, test set)

Reference box per run: accuracy >= ERM - 0.05, DP <= ERM's attacked DP. 0 = no gain over ERM; 1 = zero gap at
ERM's accuracy. Values above 1 occur when a configuration is MORE accurate than ERM on that run.

## ACS Income (10 complete runs)

| Method | Mean HV | SD |
| --- | --- | --- |
| ERM | 0.0000 | 0.0000 |
| Fair-Only | 0.4982 | 0.0469 |
| Robust-Only | 0.5187 | 0.0198 |
| Fair adv. training (clean + adv. loss) | 0.6836 | 0.0205 |
| Fair adv. training (Chai & Wang 2023) | 0.7094 | 0.0320 |
| TRI-FRAME | 0.7251 | 0.0279 |

| TRI-FRAME minus | Mean diff | d_z | p (Holm) |
| --- | --- | --- | --- |
| Fair-Only | +0.2269 | 4.11 | 0.0000 |
| Robust-Only | +0.2065 | 7.80 | 0.0000 |
| Fair adv. training (clean + adv. loss) | +0.0416 | 2.21 | 0.0001 |
| Fair adv. training (Chai & Wang 2023) | +0.0157 | 0.63 | 0.0759 |

## Adult (10 complete runs)

| Method | Mean HV | SD |
| --- | --- | --- |
| ERM | 0.0000 | 0.0000 |
| Fair-Only | 0.4721 | 0.0260 |
| Robust-Only | 0.4879 | 0.0406 |
| Fair adv. training (clean + adv. loss) | 0.7686 | 0.0223 |
| Fair adv. training (Chai & Wang 2023) | 0.7664 | 0.0304 |
| TRI-FRAME | 0.8057 | 0.0299 |

| TRI-FRAME minus | Mean diff | d_z | p (Holm) |
| --- | --- | --- | --- |
| Fair-Only | +0.3336 | 8.33 | 0.0000 |
| Robust-Only | +0.3178 | 9.86 | 0.0000 |
| Fair adv. training (clean + adv. loss) | +0.0371 | 1.40 | 0.0020 |
| Fair adv. training (Chai & Wang 2023) | +0.0393 | 1.51 | 0.0020 |

## COMPAS (10 complete runs)

| Method | Mean HV | SD |
| --- | --- | --- |
| ERM | 0.0000 | 0.0000 |
| Fair-Only | 0.0152 | 0.0405 |
| Robust-Only | 0.5003 | 0.1101 |
| Fair adv. training (clean + adv. loss) | 0.4981 | 0.0789 |
| Fair adv. training (Chai & Wang 2023) | 0.3149 | 0.1220 |
| TRI-FRAME | 0.4829 | 0.0877 |

| TRI-FRAME minus | Mean diff | d_z | p (Holm) |
| --- | --- | --- | --- |
| Fair-Only | +0.4677 | 5.41 | 0.0000 |
| Robust-Only | -0.0174 | -0.29 | 0.6924 |
| Fair adv. training (clean + adv. loss) | -0.0152 | -0.31 | 0.6924 |
| Fair adv. training (Chai & Wang 2023) | +0.1680 | 2.35 | 0.0001 |

## South German Credit (25 complete runs)

| Method | Mean HV | SD |
| --- | --- | --- |
| ERM | 0.0000 | 0.0000 |
| Fair-Only | 0.8428 | 0.2960 |
| Robust-Only | 0.5998 | 0.3641 |
| Fair adv. training (clean + adv. loss) | 0.9904 | 0.2856 |
| Fair adv. training (Chai & Wang 2023) | 1.0086 | 0.3946 |
| TRI-FRAME | 0.9734 | 0.3731 |

| TRI-FRAME minus | Mean diff | d_z | p (Holm) |
| --- | --- | --- | --- |
| Fair-Only | +0.1306 | 0.55 | 0.0332 |
| Robust-Only | +0.3736 | 0.99 | 0.0002 |
| Fair adv. training (clean + adv. loss) | -0.0171 | -0.08 | 0.8933 |
| Fair adv. training (Chai & Wang 2023) | -0.0352 | -0.15 | 0.8933 |
