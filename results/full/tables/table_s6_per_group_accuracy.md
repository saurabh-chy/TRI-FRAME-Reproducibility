| Dataset | Method | Condition | Accuracy a=0 | Accuracy a=1 | Gap (a=1 minus a=0) |
| --- | --- | --- | --- | --- | --- |
| Adult | ERM | clean | 0.9267 ± 0.0030 | 0.8169 ± 0.0034 | -0.1098 |
| Adult | ERM | PGD accuracy attack | 0.8627 ± 0.0084 | 0.6645 ± 0.0145 | -0.1982 |
| Adult | ERM | group-targeted fairness attack | 0.9267 ± 0.0030 | 0.7729 ± 0.0116 | -0.1538 |
| Adult | Fair-Only | clean | 0.9206 ± 0.0034 | 0.8112 ± 0.0035 | -0.1094 |
| Adult | Fair-Only | PGD accuracy attack | 0.8439 ± 0.0098 | 0.6763 ± 0.0145 | -0.1676 |
| Adult | Fair-Only | group-targeted fairness attack | 0.9206 ± 0.0034 | 0.7946 ± 0.0083 | -0.1261 |
| Adult | Robust-Only | clean | 0.9247 ± 0.0037 | 0.8098 ± 0.0036 | -0.1149 |
| Adult | Robust-Only | PGD accuracy attack | 0.9163 ± 0.0034 | 0.7928 ± 0.0040 | -0.1236 |
| Adult | Robust-Only | group-targeted fairness attack | 0.9247 ± 0.0037 | 0.8096 ± 0.0031 | -0.1151 |
| Adult | Fair adv. training (clean + adv. loss) | clean | 0.9227 ± 0.0035 | 0.8133 ± 0.0042 | -0.1094 |
| Adult | Fair adv. training (clean + adv. loss) | PGD accuracy attack | 0.9042 ± 0.0033 | 0.7676 ± 0.0044 | -0.1365 |
| Adult | Fair adv. training (clean + adv. loss) | group-targeted fairness attack | 0.9227 ± 0.0035 | 0.8128 ± 0.0040 | -0.1099 |
| Adult | Fair adv. training (Chai & Wang 2023) | clean | 0.9236 ± 0.0038 | 0.8066 ± 0.0044 | -0.1171 |
| Adult | Fair adv. training (Chai & Wang 2023) | PGD accuracy attack | 0.9174 ± 0.0035 | 0.7945 ± 0.0044 | -0.1229 |
| Adult | Fair adv. training (Chai & Wang 2023) | group-targeted fairness attack | 0.9236 ± 0.0038 | 0.8076 ± 0.0038 | -0.1160 |
| Adult | TRI-FRAME | clean | 0.9220 ± 0.0037 | 0.8093 ± 0.0047 | -0.1127 |
| Adult | TRI-FRAME | PGD accuracy attack | 0.9128 ± 0.0040 | 0.7875 ± 0.0043 | -0.1254 |
| Adult | TRI-FRAME | group-targeted fairness attack | 0.9220 ± 0.0037 | 0.8083 ± 0.0048 | -0.1138 |
| ACS Income | ERM | clean | 0.8261 ± 0.0031 | 0.8111 ± 0.0035 | -0.0150 |
| ACS Income | ERM | PGD accuracy attack | 0.7139 ± 0.0058 | 0.7124 ± 0.0077 | -0.0015 |
| ACS Income | ERM | group-targeted fairness attack | 0.8135 ± 0.0043 | 0.8111 ± 0.0035 | -0.0024 |
| ACS Income | Fair-Only | clean | 0.8189 ± 0.0038 | 0.8057 ± 0.0036 | -0.0133 |
| ACS Income | Fair-Only | PGD accuracy attack | 0.7176 ± 0.0045 | 0.7210 ± 0.0062 | +0.0034 |
| ACS Income | Fair-Only | group-targeted fairness attack | 0.8216 ± 0.0042 | 0.8057 ± 0.0036 | -0.0160 |
| ACS Income | Robust-Only | clean | 0.8206 ± 0.0027 | 0.8081 ± 0.0042 | -0.0125 |
| ACS Income | Robust-Only | PGD accuracy attack | 0.7663 ± 0.0029 | 0.7595 ± 0.0048 | -0.0068 |
| ACS Income | Robust-Only | group-targeted fairness attack | 0.8216 ± 0.0041 | 0.8081 ± 0.0042 | -0.0135 |
| ACS Income | Fair adv. training (clean + adv. loss) | clean | 0.8217 ± 0.0029 | 0.8075 ± 0.0040 | -0.0143 |
| ACS Income | Fair adv. training (clean + adv. loss) | PGD accuracy attack | 0.7458 ± 0.0037 | 0.7428 ± 0.0060 | -0.0031 |
| ACS Income | Fair adv. training (clean + adv. loss) | group-targeted fairness attack | 0.8234 ± 0.0035 | 0.8075 ± 0.0040 | -0.0160 |
| ACS Income | Fair adv. training (Chai & Wang 2023) | clean | 0.8157 ± 0.0035 | 0.8044 ± 0.0050 | -0.0113 |
| ACS Income | Fair adv. training (Chai & Wang 2023) | PGD accuracy attack | 0.7839 ± 0.0055 | 0.7764 ± 0.0061 | -0.0075 |
| ACS Income | Fair adv. training (Chai & Wang 2023) | group-targeted fairness attack | 0.8188 ± 0.0034 | 0.8044 ± 0.0050 | -0.0144 |
| ACS Income | TRI-FRAME | clean | 0.8212 ± 0.0032 | 0.8070 ± 0.0045 | -0.0142 |
| ACS Income | TRI-FRAME | PGD accuracy attack | 0.7620 ± 0.0037 | 0.7566 ± 0.0061 | -0.0054 |
| ACS Income | TRI-FRAME | group-targeted fairness attack | 0.8240 ± 0.0039 | 0.8070 ± 0.0045 | -0.0170 |
| COMPAS | ERM | clean | 0.6802 ± 0.0074 | 0.6729 ± 0.0250 | -0.0072 |
| COMPAS | ERM | PGD accuracy attack | 0.3669 ± 0.0443 | 0.3803 ± 0.0592 | +0.0134 |
| COMPAS | ERM | group-targeted fairness attack | 0.5910 ± 0.0193 | 0.6729 ± 0.0250 | +0.0819 |
| COMPAS | Fair-Only | clean | 0.6786 ± 0.0064 | 0.6734 ± 0.0261 | -0.0052 |
| COMPAS | Fair-Only | PGD accuracy attack | 0.3616 ± 0.0448 | 0.3772 ± 0.0589 | +0.0156 |
| COMPAS | Fair-Only | group-targeted fairness attack | 0.5906 ± 0.0199 | 0.6734 ± 0.0261 | +0.0828 |
| COMPAS | Robust-Only | clean | 0.6743 ± 0.0151 | 0.6724 ± 0.0198 | -0.0019 |
| COMPAS | Robust-Only | PGD accuracy attack | 0.5498 ± 0.0178 | 0.5722 ± 0.0258 | +0.0224 |
| COMPAS | Robust-Only | group-targeted fairness attack | 0.6617 ± 0.0141 | 0.6724 ± 0.0198 | +0.0107 |
| COMPAS | Fair adv. training (clean + adv. loss) | clean | 0.6488 ± 0.0233 | 0.6665 ± 0.0237 | +0.0177 |
| COMPAS | Fair adv. training (clean + adv. loss) | PGD accuracy attack | 0.5109 ± 0.0189 | 0.5772 ± 0.0256 | +0.0663 |
| COMPAS | Fair adv. training (clean + adv. loss) | group-targeted fairness attack | 0.6627 ± 0.0106 | 0.6665 ± 0.0237 | +0.0038 |
| COMPAS | Fair adv. training (Chai & Wang 2023) | clean | 0.6542 ± 0.0154 | 0.6551 ± 0.0204 | +0.0009 |
| COMPAS | Fair adv. training (Chai & Wang 2023) | PGD accuracy attack | 0.5890 ± 0.0206 | 0.6045 ± 0.0224 | +0.0155 |
| COMPAS | Fair adv. training (Chai & Wang 2023) | group-targeted fairness attack | 0.6608 ± 0.0141 | 0.6551 ± 0.0204 | -0.0057 |
| COMPAS | TRI-FRAME | clean | 0.6608 ± 0.0128 | 0.6710 ± 0.0224 | +0.0102 |
| COMPAS | TRI-FRAME | PGD accuracy attack | 0.5526 ± 0.0126 | 0.5960 ± 0.0232 | +0.0434 |
| COMPAS | TRI-FRAME | group-targeted fairness attack | 0.6587 ± 0.0143 | 0.6710 ± 0.0224 | +0.0123 |
| South German Credit | ERM | clean | 0.6779 ± 0.0718 | 0.7751 ± 0.0291 | +0.0972 |
| South German Credit | ERM | PGD accuracy attack | 0.5842 ± 0.0853 | 0.7240 ± 0.0272 | +0.1397 |
| South German Credit | ERM | group-targeted fairness attack | 0.6768 ± 0.0660 | 0.7738 ± 0.0286 | +0.0970 |
| South German Credit | Fair-Only | clean | 0.6442 ± 0.0450 | 0.7701 ± 0.0261 | +0.1259 |
| South German Credit | Fair-Only | PGD accuracy attack | 0.5863 ± 0.0524 | 0.7225 ± 0.0282 | +0.1362 |
| South German Credit | Fair-Only | group-targeted fairness attack | 0.6379 ± 0.0533 | 0.7709 ± 0.0261 | +0.1330 |
| South German Credit | Robust-Only | clean | 0.6642 ± 0.0702 | 0.7805 ± 0.0285 | +0.1163 |
| South German Credit | Robust-Only | PGD accuracy attack | 0.6421 ± 0.0688 | 0.7585 ± 0.0299 | +0.1164 |
| South German Credit | Robust-Only | group-targeted fairness attack | 0.6747 ± 0.0696 | 0.7810 ± 0.0294 | +0.1063 |
| South German Credit | Fair adv. training (clean + adv. loss) | clean | 0.6526 ± 0.0510 | 0.7741 ± 0.0212 | +0.1214 |
| South German Credit | Fair adv. training (clean + adv. loss) | PGD accuracy attack | 0.6168 ± 0.0487 | 0.7477 ± 0.0247 | +0.1308 |
| South German Credit | Fair adv. training (clean + adv. loss) | group-targeted fairness attack | 0.6421 ± 0.0532 | 0.7758 ± 0.0216 | +0.1337 |
| South German Credit | Fair adv. training (Chai & Wang 2023) | clean | 0.6284 ± 0.0404 | 0.7637 ± 0.0216 | +0.1353 |
| South German Credit | Fair adv. training (Chai & Wang 2023) | PGD accuracy attack | 0.6147 ± 0.0401 | 0.7528 ± 0.0216 | +0.1381 |
| South German Credit | Fair adv. training (Chai & Wang 2023) | group-targeted fairness attack | 0.6263 ± 0.0436 | 0.7637 ± 0.0216 | +0.1374 |
| South German Credit | TRI-FRAME | clean | 0.6232 ± 0.0441 | 0.7738 ± 0.0212 | +0.1507 |
| South German Credit | TRI-FRAME | PGD accuracy attack | 0.6158 ± 0.0443 | 0.7662 ± 0.0201 | +0.1504 |
| South German Credit | TRI-FRAME | group-targeted fairness attack | 0.6189 ± 0.0457 | 0.7751 ± 0.0209 | +0.1561 |