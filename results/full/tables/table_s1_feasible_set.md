| Dataset | Feature | Type | Role | Budget (raw units, 1x) | Budget (SD units, 1x) | Valid range (train) |
| --- | --- | --- | --- | --- | --- | --- |
| ACS Income | AGEP | numeric, integer | immutable |  |  | [17, 94] |
| ACS Income | SCHL | numeric, integer | mutable | 1 | 0.253 | [1, 24] |
| ACS Income | WKHP | numeric, integer | mutable | 4 | 0.308 | [1, 99] |
| ACS Income | COW | categorical (8 levels) | frozen |  |  |  |
| ACS Income | MAR | categorical (5 levels) | frozen |  |  |  |
| ACS Income | OCCP_GRP | categorical (99 levels) | frozen |  |  |  |
| ACS Income | POBP_GRP | categorical (6 levels) | frozen |  |  |  |
| ACS Income | RELP | categorical (18 levels) | frozen |  |  |  |
| ACS Income | RAC1P | categorical (9 levels) | frozen |  |  |  |
| ACS Income | sensitive attribute | SEX (1 = male) | not a model input |  |  |  |
| Adult | age | numeric, integer | immutable |  |  | [17, 90] |
| Adult | education-num | numeric, integer | mutable | 1 | 0.39 | [1, 16] |
| Adult | capital-gain | numeric, integer | mutable | 2000 | 0.27 | [0, 99999] |
| Adult | capital-loss | numeric, integer | mutable | 200 | 0.497 | [0, 3900] |
| Adult | hours-per-week | numeric, integer | mutable | 4 | 0.334 | [1, 99] |
| Adult | workclass | categorical (7 levels) | frozen |  |  |  |
| Adult | marital-status | categorical (7 levels) | frozen |  |  |  |
| Adult | occupation | categorical (14 levels) | frozen |  |  |  |
| Adult | relationship | categorical (6 levels) | frozen |  |  |  |
| Adult | race | categorical (5 levels) | frozen |  |  |  |
| Adult | native-country | categorical (40 levels) | frozen |  |  |  |
| Adult | sensitive attribute | sex (1 = male) | not a model input |  |  |  |
| COMPAS | age | numeric, integer | immutable |  |  | [18, 79] |
| COMPAS | priors_count | numeric, integer | mutable | 1 | 0.206 | [0, 37] |
| COMPAS | juv_fel_count | numeric, integer | mutable | 1 | 2.26 | [0, 8] |
| COMPAS | juv_misd_count | numeric, integer | mutable | 1 | 2.079 | [0, 12] |
| COMPAS | juv_other_count | numeric, integer | mutable | 1 | 2.238 | [0, 6] |
| COMPAS | sex | categorical (2 levels) | frozen |  |  |  |
| COMPAS | c_charge_degree | categorical (2 levels) | frozen |  |  |  |
| COMPAS | sensitive attribute | race (1 = Caucasian, 0 = African-American) | not a model input |  |  |  |
| South German Credit | duration | numeric, integer | mutable | 6 | 0.485 | [4, 60] |
| South German Credit | amount | numeric, integer | mutable | 500 | 0.177 | [250, 15945] |
| South German Credit | installment_rate | numeric, integer | immutable |  |  | [1, 4] |
| South German Credit | present_residence | numeric, integer | immutable |  |  | [1, 4] |
| South German Credit | number_credits | numeric, integer | immutable |  |  | [1, 4] |
| South German Credit | status | categorical (4 levels) | frozen |  |  |  |
| South German Credit | credit_history | categorical (5 levels) | frozen |  |  |  |
| South German Credit | purpose | categorical (10 levels) | frozen |  |  |  |
| South German Credit | savings | categorical (5 levels) | frozen |  |  |  |
| South German Credit | employment_duration | categorical (5 levels) | frozen |  |  |  |
| South German Credit | personal_status_sex | categorical (4 levels) | frozen |  |  |  |
| South German Credit | other_debtors | categorical (3 levels) | frozen |  |  |  |
| South German Credit | property | categorical (4 levels) | frozen |  |  |  |
| South German Credit | other_installment_plans | categorical (3 levels) | frozen |  |  |  |
| South German Credit | housing | categorical (3 levels) | frozen |  |  |  |
| South German Credit | job | categorical (4 levels) | frozen |  |  |  |
| South German Credit | people_liable | categorical (2 levels) | frozen |  |  |  |
| South German Credit | telephone | categorical (2 levels) | frozen |  |  |  |
| South German Credit | foreign_worker | categorical (2 levels) | frozen |  |  |  |
| South German Credit | sensitive attribute | age (1 = over 25, 0 = 25 or under) | not a model input |  |  |  |