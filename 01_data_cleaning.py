"""
01_data_cleaning.py
===================
电商数据清洗脚本

业务背景：
    原始数据来自英国某在线零售商 2010-2011 年的交易记录，包含正常订单和退货单，
    以及匿名用户（CustomerID 为空）的消费记录。清洗后的数据将用于后续 EDA、
    RFM 客户分层和 Cohort 留存分析。

输出：
    data/cleaned_data.csv  — 清洗后的全量订单数据（含退货标记）
    data/orders_for_rfm.csv — 仅含有 CustomerID 的正常消费订单（用于 RFM/Cohort）
"""

import pandas as pd
import numpy as np
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "data.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
RFM_PATH = os.path.join(BASE_DIR, "data", "orders_for_rfm.csv")


def load_raw_data(path: str) -> pd.DataFrame:
    """加载原始 CSV，指定编码避免乱码。"""
    df = pd.read_csv(path, encoding="ISO-8859-1", dtype={"CustomerID": str})
    print(f"[加载] 原始数据行数: {len(df):,}，列数: {df.shape[1]}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """打印基础质量概览。"""
    print("\n── 缺失值统计 ──")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())
    print(f"\n── 重复行: {df.duplicated().sum():,} 条 ──")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行以下清洗步骤：
    1. 删除完全重复行
    2. 解析 InvoiceDate 为 datetime
    3. 标记退货单（InvoiceNo 以 'C' 开头）
    4. 过滤单价 <= 0 的异常记录（赠品/系统内部条目）
    5. 删除 Description 为空的行（极少数，无业务价值）
    6. 规范 CustomerID 格式（去除小数点）
    7. 新增衍生字段：TotalPrice、Year、Month、Hour、DayOfWeek
    """

    # 1. 去重
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[去重] 删除 {before - len(df):,} 条重复行，剩余 {len(df):,} 条")

    # 2. 解析日期
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")

    # 3. 标记退货单（不直接删除，保留用于退货率分析）
    df["IsReturn"] = df["InvoiceNo"].astype(str).str.startswith("C")
    print(f"[退货] 退货单 {df['IsReturn'].sum():,} 条"
          f"（占比 {df['IsReturn'].mean()*100:.1f}%）")

    # 4. 过滤单价异常（负价格、0 价格）
    price_anomalies = (df["UnitPrice"] <= 0)
    df = df[~price_anomalies]
    print(f"[价格] 过滤异常单价记录 {price_anomalies.sum():,} 条")

    # 5. 删除 Description 为空的行
    desc_missing = df["Description"].isnull()
    df = df[~desc_missing]
    print(f"[描述] 删除无商品描述记录 {desc_missing.sum():,} 条")

    # 过滤非商品条目（邮费、折扣、手续费、坏账调整等）
    NON_PRODUCT_CODES = {'POST', 'DOT', 'M', 'D', 'S', 'AMAZONFEE', 
                     'CRUK', 'DCGSSGIRL', 'DCGSSBOY', 'PADS', 'B', 'm'}
    before = len(df)
    df = df[~df['StockCode'].isin(NON_PRODUCT_CODES)]
    print(f"[非商品] 过滤非商品条目 {before - len(df):,} 条")

    # 过滤 Description 为空且 UnitPrice 为 0 的库存调整行
    adj_mask = df['Description'].isna() & (df['UnitPrice'] == 0)
    df = df[~adj_mask]
    print(f"[调整行] 过滤库存调整记录 {adj_mask.sum():,} 条")

    # 6. 规范 CustomerID（去除 .0 后缀，保持字符串型）
    df["CustomerID"] = df["CustomerID"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else np.nan
    )

    # 7. 衍生字段
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    print(f"\n[完成] 清洗后数据行数: {len(df):,}")
    return df


def create_rfm_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    从清洗后的数据中过滤出用于 RFM/Cohort 分析的子集：
    - 排除退货单
    - 排除匿名用户（CustomerID 为空）
    - 排除 Quantity <= 0 的行
    """
    rfm_df = df[
        (~df["IsReturn"]) &
        (df["CustomerID"].notna()) &
        (df["Quantity"] > 0)
    ].copy()
    print(f"\n[RFM基础数据] 有 CustomerID 的正常订单: {len(rfm_df):,} 条"
          f"，涉及客户: {rfm_df['CustomerID'].nunique():,} 人")
    return rfm_df


def main():
    df = load_raw_data(RAW_PATH)
    basic_info(df)

    df_clean = clean_data(df)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\n✅ 清洗后全量数据已保存至: {CLEAN_PATH}")

    df_rfm = create_rfm_base(df_clean)
    df_rfm.to_csv(RFM_PATH, index=False)
    print(f"✅ RFM 基础数据已保存至: {RFM_PATH}")

    # 最终质量报告
    print("\n── 最终数据质量概览 ──")
    print(f"  时间范围: {df_clean['InvoiceDate'].min().date()} "
          f"~ {df_clean['InvoiceDate'].max().date()}")
    print(f"  涉及国家: {df_clean['Country'].nunique()} 个")
    print(f"  商品种类: {df_clean['StockCode'].nunique():,} 个")
    print(f"  有效客户: {df_clean['CustomerID'].nunique():,} 人（含匿名）")


if __name__ == "__main__":
    main()
