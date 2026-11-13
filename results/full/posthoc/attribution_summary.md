# Per-feature attribution: DP gap under the worst-case group-targeted attack at 1x,
# with only the named feature editable (mean over runs)

## ACS Income

| Method | Clean DP | ALL mutable features | SCHL | WKHP |
| --- | --- | --- | --- | --- |
| ERM | 0.085 | 0.182 | 0.122 | 0.144 |
| Fair-Only | 0.021 | 0.114 | 0.060 | 0.073 |
| Robust-Only | 0.078 | 0.128 | 0.094 | 0.115 |
| Fair adv. training (clean + adv. loss) | 0.041 | 0.110 | 0.067 | 0.085 |
| Fair adv. training (Chai & Wang 2023) | 0.059 | 0.097 | 0.064 | 0.093 |
| TRI-FRAME | 0.047 | 0.102 | 0.067 | 0.085 |

## Adult

| Method | Clean DP | ALL mutable features | capital-gain | capital-loss | education-num | hours-per-week |
| --- | --- | --- | --- | --- | --- | --- |
| ERM | 0.182 | 0.365 | 0.278 | 0.191 | 0.222 | 0.199 |
| Fair-Only | 0.094 | 0.265 | 0.180 | 0.106 | 0.133 | 0.109 |
| Robust-Only | 0.182 | 0.197 | 0.192 | 0.184 | 0.185 | 0.183 |
| Fair adv. training (clean + adv. loss) | 0.107 | 0.148 | 0.124 | 0.112 | 0.125 | 0.111 |
| Fair adv. training (Chai & Wang 2023) | 0.149 | 0.161 | 0.158 | 0.151 | 0.149 | 0.149 |
| TRI-FRAME | 0.115 | 0.135 | 0.128 | 0.119 | 0.119 | 0.117 |

## COMPAS

| Method | Clean DP | ALL mutable features | juv_fel_count | juv_misd_count | juv_other_count | priors_count |
| --- | --- | --- | --- | --- | --- | --- |
| ERM | 0.238 | 0.603 | 0.411 | 0.357 | 0.389 | 0.357 |
| Fair-Only | 0.237 | 0.606 | 0.409 | 0.357 | 0.395 | 0.357 |
| Robust-Only | 0.258 | 0.385 | 0.280 | 0.280 | 0.287 | 0.353 |
| Fair adv. training (clean + adv. loss) | 0.174 | 0.321 | 0.205 | 0.202 | 0.222 | 0.265 |
| Fair adv. training (Chai & Wang 2023) | 0.253 | 0.327 | 0.264 | 0.262 | 0.262 | 0.305 |
| TRI-FRAME | 0.233 | 0.339 | 0.256 | 0.251 | 0.262 | 0.320 |

## South German Credit

| Method | Clean DP | ALL mutable features | amount | duration |
| --- | --- | --- | --- | --- |
| ERM | 0.101 | 0.194 | 0.115 | 0.172 |
| Fair-Only | 0.038 | 0.076 | 0.044 | 0.070 |
| Robust-Only | 0.104 | 0.138 | 0.122 | 0.126 |
| Fair adv. training (clean + adv. loss) | 0.041 | 0.065 | 0.049 | 0.058 |
| Fair adv. training (Chai & Wang 2023) | 0.041 | 0.051 | 0.047 | 0.046 |
| TRI-FRAME | 0.042 | 0.049 | 0.046 | 0.047 |
