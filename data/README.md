# Data

No data are redistributed here. Download the four public datasets and put the files (zipped or unzipped)
in a folder named `download` next to the `experiments` folder. The pipeline searches `download/`
recursively and extracts zip files itself.

| Dataset | File(s) the pipeline looks for | Source |
|---|---|---|
| Adult | `adult.data`, `adult.test` | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/2/adult (Becker & Kohavi, 1996) |
| ACS Income (California, 2018) | `psam_p06.csv` (inside `csv_pca.zip`) | US Census Bureau, ACS 2018 1-Year PUMS person file for California: https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip. If absent, the script downloads it with `folktables` (Ding et al., 2021). |
| COMPAS | `compas-scores-two-years.csv` | ProPublica, https://github.com/propublica/compas-analysis (Larson et al., 2016) |
| South German Credit | `SouthGermanCredit.asc` (inside the UCI zip) | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/522/south+german+credit (Grömping, 2019) |

Preprocessing (filters, encodings, sensitive attributes, feasible sets) is done by the pipeline and is
documented in Section S1 and Table S1 of the supplementary material.
