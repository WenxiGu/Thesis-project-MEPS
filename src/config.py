
from pathlib import Path

# 项目根目录（假设你在项目根目录运行 notebook / 脚本）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据路径
RAW_DATA_PATH = PROJECT_ROOT / "data" / "h252.xlsx"

# 输出路径
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "meps_panel27_processed.parquet"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# 随机种子
RANDOM_SEED = 42

# ----- Target variable names -----

# 原始总费用（回归目标的原始版本）
REG_TARGET_TOTEXPY2_RAW = "TOTEXPY2"
REG_BASELINE_TOTEXPY1   = "TOTEXPY1"   # Year 1 baseline cost（重要特征）

# 建模时真正用的回归目标：log(1 + TOTEXPY2)
REG_TARGET_TOTEXPY2_LOG = "LOG_TOTEXPY2"

# 高费用标签（例如基于 TOTEXPY2 的 top 10% 定义）
CLASS_TARGET_HIGHCOST_Y2 = "HIGHCOST_Y2"

# 急诊 / 住院标签（Phase II 会从次数变量构造）
CLASS_TARGET_ANY_ED_Y2 = "ANY_ED_Y2"   # 1 if any ED visit in Y2
CLASS_TARGET_ANY_IP_Y2 = "ANY_IP_Y2"   # 1 if any inpatient stay in Y2

# 你也可以保留原始次数变量名字作为参考
ED_COUNT_Y1 = "ERTOTY1"
ED_COUNT_Y2 = "ERTOTY2"
IP_COUNT_Y1 = "IPDISY1"
IP_COUNT_Y2 = "IPDISY2"

# 权重变量（MEPS 纵向权重）
WEIGHT_COL = "LONGWT"
