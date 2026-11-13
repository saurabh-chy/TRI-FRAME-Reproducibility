| Dataset | Method | Selected |
| --- | --- | --- |
| ACS Income | ERM | {} |
| ACS Income | Fair-Only | {"lam_f": 0.547722557505166} |
| ACS Income | Robust-Only | {"lam_r": 0.1, "train_eps": 2.0} |
| ACS Income | Fair adv. training (clean + adv. loss) | {"lam_f": 0.3, "lam_r": 0.1} |
| ACS Income | Fair adv. training (Chai & Wang 2023) | {"lam_f": 0.1, "train_eps": 0.5, "adv_only": true} |
| ACS Income | TRI-FRAME | {"lam_f": 0.1, "lam_r": 0.1, "lam_fa": 0.1} |
| Adult | ERM | {} |
| Adult | Fair-Only | {"lam_f": 0.2649724883447741} |
| Adult | Robust-Only | {"lam_r": 0.5477225575051661, "train_eps": 2.0} |
| Adult | Fair adv. training (clean + adv. loss) | {"lam_f": 0.3, "lam_r": 0.3} |
| Adult | Fair adv. training (Chai & Wang 2023) | {"lam_f": 0.1, "train_eps": 1.0, "adv_only": true} |
| Adult | TRI-FRAME | {"lam_f": 0.1, "lam_r": 1.0, "lam_fa": 0.3} |
| COMPAS | ERM | {} |
| COMPAS | Fair-Only | {"lam_f": 0.03} |
| COMPAS | Robust-Only | {"lam_r": 0.5477225575051661, "train_eps": 0.5} |
| COMPAS | Fair adv. training (clean + adv. loss) | {"lam_f": 1.0, "lam_r": 0.3} |
| COMPAS | Fair adv. training (Chai & Wang 2023) | {"lam_f": 0.1, "train_eps": 0.5, "adv_only": true} |
| COMPAS | TRI-FRAME | {"lam_f": 0.3, "lam_r": 0.1, "lam_fa": 0.3} |
| South German Credit | ERM | {} |
| South German Credit | Fair-Only | {"lam_f": 1.1321930132219953} |
| South German Credit | Robust-Only | {"lam_r": 0.5477225575051661, "train_eps": 1.0} |
| South German Credit | Fair adv. training (clean + adv. loss) | {"lam_f": 1.0, "lam_r": 0.3} |
| South German Credit | Fair adv. training (Chai & Wang 2023) | {"lam_f": 1.0, "train_eps": 0.5, "adv_only": true} |
| South German Credit | TRI-FRAME | {"lam_f": 0.1, "lam_r": 0.3, "lam_fa": 1.0} |