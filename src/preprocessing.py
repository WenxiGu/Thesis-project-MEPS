
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

from .config import WEIGHT_COL



# 1. 识别样本、权重相关
CORE_ID_VARS = [
    "DUID",      # dwelling unit ID 
    "PID",       # person ID within household
    "DUPERSID",  # unique person ID 个人唯一 ID，用来在 panel、中间文件、event 文件里合并。
    "PANEL",     # panel number (27)
    "YEARIND",   # in both years / only 2022 / only 2023 是否在两个年份都在样本框中；=1 表示 2022 和 2023 都在，是纵向分析推荐子样本。
    "ALL5RDS",   # =1 if in-scope and responded all 5 rounds
    "DIED",      # =1 if died during 2-year period
    "INST",      # ever institutionalized during panel 是否在两年中有过住院机构 / long-term care 的 institutionalization。
    "MILITARY",  # ever active duty military 是否在两年中有 active duty military（在军队）。
    "ENTRSRVY",  # entered survey late
    "LEFTUS",    # left US during panel
    "OTHER",     # other special status 一些特殊 sample 状态（中间进入调查、离开美国等）。
]

#“We restricted the sample to individuals present from the beginning of the panel (ENTRSRVY=0), with data collected in all five rounds (ALL5RDS=1) …”


# 抽样设计 & 权重
CORE_DESIGN_VARS = [
    WEIGHT_COL,  # LONGWT: longitudinal person weight
    "LSAQWT",    # SAQ longitudinal weight (for SAQ vars, if用到)
    "VARSTR",    # stratum 抽样层 这个人所在的“大抽样层”
    "VARPSU",    # PSU Primary Sampling Unit（PSU，抽样单位） 这个人所在的“具体抽样小组”
]

# VARSTR, VARPSU：抽样分层和 PSU，用于 survey-weighted 估计和方差计算。

# SAQ = Self-Administered Questionnaire（自填问卷，成人在 Round 2 & 4 做）

# 含义：

# 这只对 完成了 SAQ 的那部分子样本 有意义。

# 用它可以让这批“做过 SAQ 的人”在分析 SAQ 相关问题时，也能代表全国（成年人）人群。

# 什么时候用：

# 只有当你用到 SAQ 问卷里的变量时，才用 LSAQWT，例如：

# patient experience / satisfaction

# access to care

# some HRQoL / PROMs 指标

# 且通常要配合 SAQRDS24=1（R2 & R4 SAQ 完整）去定义子样本。

# 对论文的直接建议

# 主样本（Panel 27 全部人，做费用/ER/住院预测）：

# 用 LONGWT（你在 config.WEIGHT_COL 里已经设成这个就对了）。

# 描述统计 & 模型评估时，尽量在指标里传 sample_weight=LONGWT。

# 如果后面你想加一个“小章节”看 SAQ（满意度/体验）跟 high cost 的关系：

# 定义子样本：SAQRDS24 = 1 & 年龄 ≥ 18

# 那一部分的描述统计/模型，就使用 LSAQWT 作为权重。

# 现在你只做费用/利用预测的话，可以简单记住：

# 99% 情况 → 用 LONGWT；
# 用到 SAQ 问卷题目，专门做 SAQ 子样本分析 → 用 LSAQWT。


# 2. 人口学（demographics）
CORE_DEMO_VARS = [
    "AGEY1X", "AGEY2X",   # age end of year1 / year2
    "AGELSTY1", "AGELSTY2",  # last age in each year
    "SEX",        # sex
    "RACETHX",    # race/ethnicity combined
    "HISPANX",    # Hispanic indicator
    "EDUCYR",     # years of education
    "REGIONY1", "REGIONY2",  # census region year1 / year2
]

# 3. 家庭经济 & 规模 socio-economic status (SES)
CORE_SES_VARS = [
    "FAMINCY1", "FAMINCY2",   # family total income Y1/Y2
    "POVCATY1", "POVCATY2",   # income as % of poverty line (categorical)  “这家人属于哪个收入档位？”
    "POVLEVY1", "POVLEVY2",   # income–poverty level (continuous) “这家人收入是贫困线的几倍（连续比例）？”
    "FAMSZEY1", "FAMSZEY2",   # family size 年末
    "RUSIZEY1", "RUSIZEY2",   # reporting unit size 年末
]

# Socio-economic status (SES) was primarily measured at the family level using total family income (FAMINCY1, FAMINCY2) and official MEPS poverty categories (POVCATY1, POVCATY2). 
# We chose family-level rather than individual income because medical expenditures are typically financed at the household level and 
# many individuals in the sample (e.g. children, non-working spouses) have no personal earnings despite living in high-income households.

# Reporting Unit (RU) = 报告单元
# 简单理解：一起被当作一个“家庭/户”来访谈的人。

# 一个 RU 里的人共用一套问卷，由一个“家庭受访者”代表回答大部分问题；

# RU 不一定和法律意义上的 “family” 完全一样，但非常接近“同一个家庭/住户”。

#如果想用得简单一点，可以只在模型里用 FAMSZEY1 或 RUSIZEY1（选一个），另外一个留在数据里但不一定进模型。


# 4. 医保覆盖情况 Health insurance coverage
CORE_INS_VARS = [
    "INSCOVY1", "INSCOVY2",   # 是否全年 covered by any insurance（指示变量）。
    "INSURCY1", "INSURCY2",   # full-year coverage type,（如 <65 any private, <65 public only, <65 uninsured, 65+ Medicare only 等）。
    "UNINSY1", "UNINSY2",     # months uninsured in year1/year2 全年累计无保险的月数。
    "PREVCOVR", "MORECOVR",   # had previous coverage / multiple coverage indicators 过去是否有保险 / 是否有多重 coverage 的指示。
]

# 保险状态是 ER / 住院 / 高费用 最核心的预测因子之一；

# 你可以分析：保险类型不同，预测误差是否不同；

# 同时在 productization 章节里，也可以讨论“在什么 insurance 组建议用模型”。

# 5. 就业相关 Employment（只留 summary）
CORE_EMP_VARS = [
    "EVRWRKY1", "EVRWRKY2",   # ever worked during year1 / year2
    "EMPST1", "EMPST2", "EMPST3", "EMPST4", "EMPST5",  # employment status in each round
    "UNEMPY1X", "UNEMPY2X",   # months unemployed in each year
]
# 就业是 SES 的一部分，也和保险类型、收入挂钩；

# 你不需要所有 employment 细节，只要有几个 summary 指标控制 socio-economic status 即可。

# 6. 自报健康 / 心理健康
CORE_HEALTH_STATUS_VARS = [
    "RTHLTH1", "RTHLTH3", "RTHLTH5",   # perceived health status R1/R3/R5  round 1/3/5 的 自评总体健康（excellent / very good / good / fair / poor）
    "MNHLTH1", "MNHLTH3", "MNHLTH5",   # perceived mental health R1/R3/R5  对应轮次的 自评心理健康
]

# 这些是 health status 的 summary 指标，比逐项 symptom 变量更稳定；

# 你可以构造变量：

# baseline health = RTHLTH1 / MNHLTH1

# mid-panel health = RTHLTH3 / MNHLTH3

# 也可以在 EDA 里比较不同 health status 下的费用差异。




# 7. 重要慢性病指示变量（Y1/Y2） chronic conditions
CORE_CHRONIC_VARS = [
    "HIBPDXY1", "HIBPDXY2",        # high blood pressure diagnosis
    "CHDDXY1", "CHDDXY2",          # coronary heart disease diagnosis
    "STRKDXY1", "STRKDXY2",        # stroke diagnosis
    "CHOLDXY1", "CHOLDXY2",        # high cholesterol diagnosis
    "ASTHDXY1", "ASTHDXY2",        # asthma diagnosis
    "DIABDXY1_M18", "DIABDXY2_M18" # diabetes diagnosis 2022/23 
]


# 这些都是 高费用 & 住院风险 很强的预测因子；

# 数量适中（5–6 个），方便之后做一个“multi-morbidity index”（慢病个数），又不会让特征太爆炸。 所以先选一小组“六大慢病”，足够支撑一个 可解释的 baseline 模型。


# 8. 医疗利用 & 费用（Y1=2022, Y2=2023）
CORE_USE_COST_VARS = [
    # 利用：ER & 住院次数（结局 + 特征）
    "ERTOTY1", "ERTOTY2",   # total # ER visits per year  一年内急诊 (ER) 访问次数
    "IPDISY1", "IPDISY2",   # # hospital discharges per year  一年内住院出院次数 (# hospital discharges)

    # 总费用 & 总 charge
    "TOTEXPY1", "TOTEXPY2",     # total health care expenditures (all payers)  每人的总医疗支出（所有服务 + 所有支付方的实付金额）
    "TOTTCHY1", "TOTTCHY2",     # total health care charges (excl Rx)  总医疗 “charges”（账面收费，排除处方药）

    # 各 payer 付的钱（Y1 / Y2）——结构信息 + 特征
    "TOTSLFY1", "TOTSLFY2",   # total paid by self/family 
    "TOTMCRY1", "TOTMCRY2",   # total paid by Medicare 
    "TOTMCDY1", "TOTMCDY2",   # total paid by Medicaid 
    "TOTPRVY1", "TOTPRVY2",   # total paid by private insurance 
    "TOTVAY1", "TOTVAY2",     # total paid by VA/CHAMPVA 
    "TOTTRIY1", "TOTTRIY2",   # total paid by TRICARE  
    "TOTOFDY1", "TOTOFDY2",   # other federal sources 其它联邦资金
    "TOTSTLY1", "TOTSTLY2",   # other state/local sources 州 / 地方政府资金
    "TOTWCPY1", "TOTWCPY2",   # workers’ compensation 工伤保险 (workers’ comp)
    "TOTOSRY1", "TOTOSRY2",   # other sources
    "TOTPTRY1", "TOTPTRY2",   # private + TRICARE combined
    "TOTOTHY1", "TOTOTHY2",   # other payers combined 
]

# 为什么整组保留？

# 对建模：你可以选一部分作为 feature（例如 Y1 各 payer 的 expenditure，用 log1p 转换）；

# 对论文 EDA / policy 讨论：

# 可以拆开看 “high-cost patients 中，哪些 payer 在买单”；

# 也方便你做图：按收入/保险组比较不同 payer 的成本结构。


# 汇总所有核心变量
CORE_VARS = (
    CORE_ID_VARS
    + CORE_DESIGN_VARS
    + CORE_DEMO_VARS
    + CORE_SES_VARS
    + CORE_INS_VARS
    + CORE_EMP_VARS
    + CORE_HEALTH_STATUS_VARS
    + CORE_CHRONIC_VARS
    + CORE_USE_COST_VARS
)

def select_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 2600+ 列中，挑出论文/建模要用的核心变量。
    其它列在 df_raw 里保留，但不进入后续 EDA / modeling。

    返回:
        df_sel: 只包含 CORE_VARS 中、且真实存在于 df 的列。
    """
    existing = [c for c in CORE_VARS if c in df.columns]
    missing = [c for c in CORE_VARS if c not in df.columns]

    if missing:
        print("Warning: these core vars not found in df and will be skipped:")
        print(missing)

    return df[existing].copy()



# MEPS special missing codes per AHRQ/IPUMS docs
# See: "Changes to MEPS Missing Data Codes" user note
MEPS_MISSING_CODES = [-1, -2, -3, -7, -8, -9, -13, -15]


def replace_special_missing(
    df: pd.DataFrame,
    codes: list[int] = MEPS_MISSING_CODES,
) -> pd.DataFrame:
    """
    Replace MEPS special missing codes with NaN.

    Parameters
    ----------
    df : DataFrame
        Input dataframe (numeric + non-numeric).
    codes : list of int
        MEPS missing codes to replace, e.g. [-1, -2, -3, -7, -8, -9, -13, -15].

    Returns
    -------
    df_clean : DataFrame
        Dataframe with these codes replaced by NaN.
    """
    df_clean = df.copy()
    # 只对数值列做替换，避免误伤字符串
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].replace(codes, np.nan)
    return df_clean



#简单的 “负金额 → NaN” 清理

EXPENDITURE_COLS = [
    "TOTEXPY1", "TOTEXPY2",
    "TOTTCHY1", "TOTTCHY2",
    "TOTSLFY1", "TOTSLFY2",
    "TOTMCRY1", "TOTMCRY2",
    "TOTMCDY1", "TOTMCDY2",
    "TOTPRVY1", "TOTPRVY2",
    "TOTVAY1",  "TOTVAY2",
    "TOTTRIY1", "TOTTRIY2",
    "TOTOFDY1", "TOTOFDY2",
    "TOTSTLY1", "TOTSTLY2",
    "TOTWCPY1", "TOTWCPY2",
    "TOTOSRY1", "TOTOSRY2",
    "TOTPTRY1", "TOTPTRY2",
    "TOTOTHY1", "TOTOTHY2",
]


def clean_negative_expenditures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expenditure variables are non-negative.
    Any negative values (after missing codes replacement)
    are set to NaN.
    """
    df_clean = df.copy()
    for col in EXPENDITURE_COLS:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan
    return df_clean




def preprocess_meps(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    High-level preprocessing pipeline (light / default version):

    1. Select core variables.
    2. Optionally restrict to complete panel (ALL5RDS == 1 & YEARIND == 1).
    3. Replace MEPS special missing codes with NaN.
    4. Clean negative expenditure values.
    5. Drop coverage flags that are almost entirely missing (PREVCOVR, MORECOVR).

    More aggressive steps (winsorization, dropping extreme outliers)
    are handled later in feature engineering / modeling.
    """
    df = select_core_columns(df_raw).copy()

    
    if "ALL5RDS" in df.columns:
        df = df[df["ALL5RDS"] == 1]
    if "YEARIND" in df.columns:
        df = df[df["YEARIND"] == 1]

    # 替换 special missing code 为 NaN
    df = replace_special_missing(df)

    # 金额负值 → NaN
    df = clean_negative_expenditures(df)
       
     # 丢掉几乎全缺的 coverage flags（以后不当特征用）
    for col in ["PREVCOVR", "MORECOVR"]:
        if col in df.columns:
            # 也可以直接 drop，不检查；这里多一行只是更安全
            if df[col].isna().mean() > 0.8:
                df = df.drop(columns=col)

    return df






