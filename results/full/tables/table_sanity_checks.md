| Dataset | Method | Acc falls with eps | PGD >= FGSM | White-box >= transfer (acc) | White-box >= transfer (DP) | Empirical soft DP <= certified | Empirical DP <= certified | Empirical DP (attack) | Certified DP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACS Income | ERM | pass | pass | pass | pass | pass | pass | 0.2609 | 0.4835 |
| ACS Income | Fair-Only | pass | pass | pass | pass | pass | pass | 0.1836 | 0.4104 |
| ACS Income | Robust-Only | pass | pass | pass | pass | pass | pass | 0.1706 | 0.4251 |
| ACS Income | Fair adv. training (clean + adv. loss) | pass | pass | pass | pass | pass | pass | 0.1651 | 0.383 |
| ACS Income | Fair adv. training (Chai & Wang 2023) | pass | pass | pass | pass | pass | pass | 0.1144 | 0.4036 |
| ACS Income | TRI-FRAME | pass | pass | pass | pass | pass | pass | 0.1446 | 0.389 |
| Adult | ERM | pass | pass | pass | pass | pass | pass | 0.3896 | 0.6863 |
| Adult | Fair-Only | pass | pass | pass | pass | pass | pass | 0.2907 | 0.6482 |
| Adult | Robust-Only | pass | pass | pass | pass | pass | pass | 0.2028 | 0.5476 |
| Adult | Fair adv. training (clean + adv. loss) | pass | pass | pass | pass | pass | pass | 0.1616 | 0.6142 |
| Adult | Fair adv. training (Chai & Wang 2023) | pass | pass | pass | pass | pass | pass | 0.1655 | 0.4487 |
| Adult | TRI-FRAME | pass | pass | pass | pass | pass | pass | 0.1406 | 0.5211 |
| COMPAS | ERM | pass | pass | pass | pass | pass | pass | 0.6978 | 1.0 |
| COMPAS | Fair-Only | pass | pass | pass | pass | pass | pass | 0.7028 | 1.0 |
| COMPAS | Robust-Only | pass | pass | pass | pass | pass | pass | 0.4469 | 1.0 |
| COMPAS | Fair adv. training (clean + adv. loss) | pass | pass | pass | pass | pass | pass | 0.3916 | 1.0 |
| COMPAS | Fair adv. training (Chai & Wang 2023) | pass | pass | pass | pass | pass | pass | 0.3639 | 1.0 |
| COMPAS | TRI-FRAME | pass | pass | pass | pass | pass | pass | 0.3933 | 1.0 |
| South German Credit | ERM | pass | pass | pass | pass | pass | pass | 0.233 | 0.5016 |
| South German Credit | Fair-Only | pass | pass | pass | pass | pass | pass | 0.1076 | 0.3799 |
| South German Credit | Robust-Only | pass | pass | pass | pass | pass | pass | 0.1497 | 0.4607 |
| South German Credit | Fair adv. training (clean + adv. loss) | pass | pass | pass | pass | pass | pass | 0.0729 | 0.3731 |
| South German Credit | Fair adv. training (Chai & Wang 2023) | pass | pass | pass | pass | pass | pass | 0.0509 | 0.3264 |
| South German Credit | TRI-FRAME | pass | pass | pass | pass | pass | pass | 0.0484 | 0.331 |