# Data Directory

This directory is reserved for local MEPS data files used to reproduce the thesis
analysis and run the Streamlit MVP.

The repository does not redistribute raw MEPS public-use microdata or row-level derived
datasets. Download the official file directly from AHRQ/MEPS:

- Dataset page: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252
- Documentation: https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252doc.shtml
- Codebook: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H252&sortBy=Start

Recommended local layout:

```text
data/
  h252.xlsx          # official HC-252 Excel file downloaded from AHRQ/MEPS
  df_pre.parquet    # generated locally by preprocessing
  df_feat.parquet   # generated locally by feature engineering; used by the app demo mode
```

These files are intentionally ignored by git:

```text
data/*.xlsx
data/*.xls
data/*.zip
data/*.csv
data/*.parquet
data/*.dta
data/*.sas7bdat
```

Users are responsible for complying with the MEPS/AHRQ Data Use Agreement.
