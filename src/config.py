
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

# 目标变量名称（先写你大概率会用的，之后可以再改）
REG_TARGET_TOTEXPY2 = "TOTEXPY2"   # 回归：第2年总费用
REG_BASELINE_TOTEXPY1 = "TOTEXPY1" # 基线费用（常用特征）

# 例子：分类任务的标签
CLASS_TARGET_ER_Y2 = "ER_Y2"       # 你之后可以从 ER 总次数构造 0/1 变量
CLASS_TARGET_IP_Y2 = "IPDISY2"     # 是否住院（0/1 或次数）

# 权重变量（MEPS 纵向权重）
WEIGHT_COL = "LONGWT"
