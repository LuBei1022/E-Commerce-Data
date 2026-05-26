"""
04_cohort_analysis.py
=====================
用户 Cohort 留存分析

业务背景：
    Cohort 分析把"同一个月首次购买"的客户归为一个队列（Cohort），
    然后追踪这批客户在后续每个月的回购率。这张留存矩阵能直观回答：
      - 新用户的次月留存率是多少？
      - 留存随时间如何衰减？
      - 哪些月份获取的新用户质量更高？

    这是增长/用户运营岗位的必备分析能力，也是面试中常被问到的技能点。

输出：
    outputs/charts/09_cohort_heatmap.png   — 留存率热力图
    outputs/charts/10_cohort_retention.png — 前3个月留存折线对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "orders_for_rfm.csv")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"[加载] {len(df):,} 条订单")
    return df


def build_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建留存矩阵：
    1. 找出每位客户的首次购买月份（Cohort Month）
    2. 计算每笔订单相对于首购月的月份偏移（Cohort Index）
    3. 统计每个 Cohort × Index 组合的唯一客户数
    4. 除以各队列的初始客户数，得到留存率
    """
    # 订单月份（用 Period 保证对比精确）
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")

    # 每位客户的首购月
    first_purchase = (
        df.groupby("CustomerID")["OrderMonth"]
        .min()
        .rename("CohortMonth")
    )
    df = df.join(first_purchase, on="CustomerID")

    # 月份偏移（0 = 首购月）
    df["CohortIndex"] = (
        df["OrderMonth"].dt.to_timestamp() -
        df["CohortMonth"].dt.to_timestamp()
    ).dt.days // 30   # 粗略换算为月份偏移

    # 各 Cohort × CohortIndex 的客户数
    cohort_data = (
        df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .reset_index()
        .rename(columns={"CustomerID": "Customers"})
    )

    # 透视为矩阵
    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth", columns="CohortIndex", values="Customers"
    )

    # 留存率矩阵（第 0 列为 100%，其余除以初始人数）
    cohort_size   = cohort_pivot.iloc[:, 0]
    retention_pct = cohort_pivot.divide(cohort_size, axis=0).round(4)

    print(f"\n[Cohort] 共 {len(cohort_pivot)} 个月度队列")
    print(f"  最小队列: {cohort_size.min():.0f} 人"
          f"  最大队列: {cohort_size.max():.0f} 人")
    return retention_pct, cohort_size


# ── 图9：留存率热力图 ─────────────────────────────────────────────────────────
def plot_cohort_heatmap(retention: pd.DataFrame, cohort_size: pd.Series) -> None:
    # 只保留有足够数据的列（最多 12 个月偏移）
    max_period = min(12, retention.shape[1])
    data = retention.iloc[:, :max_period]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        data.values.astype(float),
        cmap="YlOrRd_r", vmin=0, vmax=0.5,
        aspect="auto", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 format=mticker.PercentFormatter(xmax=1))

    # 在格子中显示百分比
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6.5, color=color)

    # Y 轴：Cohort Month + 队列规模
    y_labels = [
        f"{str(m)}  (n={int(cohort_size[m]):,})"
        for m in data.index
    ]
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"M+{i}" for i in range(data.shape[1])], fontsize=8)

    ax.set_title("Monthly Cohort Retention Rate\n(% of Cohort Still Purchasing)")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort (First Purchase Month)")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "09_cohort_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图10：前几个月留存趋势对比 ────────────────────────────────────────────────
def plot_retention_curve(retention: pd.DataFrame) -> None:
    """按月份区间对比不同时期的留存曲线。"""
    max_period = min(12, retention.shape[1])
    data = retention.iloc[:, :max_period]

    # 按季度分组，取平均留存率
    cohort_months = data.index.to_timestamp()
    groups = {
        "2010 Q4": (cohort_months.year == 2010),
        "2011 Q1": (cohort_months.year == 2011) & (cohort_months.month.isin([1,2,3])),
        "2011 Q2": (cohort_months.year == 2011) & (cohort_months.month.isin([4,5,6])),
        "2011 Q3": (cohort_months.year == 2011) & (cohort_months.month.isin([7,8,9])),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    for (label, mask), color in zip(groups.items(), colors):
        if mask.sum() == 0:
            continue
        # 使用 iloc 按位置筛选，避免 Period/Timestamp 索引不匹配
        avg = data.iloc[mask].mean()
        ax.plot(avg.index, avg.values * 100,
                label=label, color=color,
                linewidth=2, marker="o", markersize=4)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("Retention Curve by Cohort Quarter")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Retention Rate (%)")
    ax.legend(frameon=False)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "10_cohort_retention.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


def print_cohort_insights(retention: pd.DataFrame) -> None:
    month1 = retention.iloc[:, 1].dropna()
    print("\n── Cohort 关键指标 ──────────────────────────────────────────────────")
    print(f"  平均 M+1 留存率: {month1.mean()*100:.1f}%")
    print(f"  最高 M+1 留存率: {month1.max()*100:.1f}%  ({month1.idxmax()})")
    print(f"  最低 M+1 留存率: {month1.min()*100:.1f}%  ({month1.idxmin()})")
    if retention.shape[1] > 3:
        month3 = retention.iloc[:, 3].dropna()
        print(f"  平均 M+3 留存率: {month3.mean()*100:.1f}%")
    print("──────────────────────────────────────────────────────────────────────")


def main():
    df = load_data(DATA_PATH)
    retention, cohort_size = build_cohort_table(df)

    print("\n[绘图] 开始生成图表...")
    plot_cohort_heatmap(retention, cohort_size)
    plot_retention_curve(retention)

    print_cohort_insights(retention)
    print("\n✅ Cohort 分析完成")


if __name__ == "__main__":
    main()
