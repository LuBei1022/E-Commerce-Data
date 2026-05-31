"""
05_anomaly_analysis.py
======================
月度异动分析（Anomaly / Fluctuation Analysis）

业务背景：
    EDA 已经展示了月度收入趋势，但"哪些月份波动异常"和"是什么驱动了异动"
    这两个问题还没有回答。本脚本通过以下步骤完成异动分析：

    1. 计算月度环比变化率，用统计方法标记异常月份
    2. 对每个异动月份，从「订单量 / 客户数 / 客单价」三个维度拆解驱动因素
    3. 找出异动月份中贡献最大的商品和国家
    4. 给出业务解释

输出：
    outputs/charts/11_mom_revenue_change.png  — 月度环比变化率（标注异动点）
    outputs/charts/12_anomaly_breakdown.png   — 异动月份驱动因素拆解
    outputs/charts/13_weekly_revenue.png      — 周度收入趋势（发现月内波动）
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
PRIMARY = "#2E86AB"
ALERT   = "#E84855"
NEUTRAL = "#b2bec3"

# 异动阈值：环比变化超过 ±30% 视为异动
ANOMALY_THRESHOLD = 0.30


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Month"]  = df["InvoiceDate"].dt.to_period("M")
    df["Week"]   = df["InvoiceDate"].dt.to_period("W")
    print(f"[加载] {len(df):,} 条订单")
    return df


# ── 核心计算：月度指标 + 环比 ─────────────────────────────────────────────────
def build_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    按月聚合三个核心指标：
    - Revenue    总收入
    - Orders     订单数（不重复发票号）
    - Customers  活跃客户数
    并计算衍生指标：
    - AvgOrderValue  客单价 = Revenue / Orders
    - MoM_Revenue    收入环比变化率
    """
    monthly = df.groupby("Month").agg(
        Revenue   =("TotalPrice",  "sum"),
        Orders    =("InvoiceNo",   "nunique"),
        Customers =("CustomerID",  "nunique"),
    ).reset_index()

    monthly["AvgOrderValue"] = monthly["Revenue"] / monthly["Orders"]
    monthly["MoM_Revenue"]   = monthly["Revenue"].pct_change()  # 环比变化率

    # 标记异动月份（排除第一个月，因为没有上月数据）
    monthly["IsAnomaly"] = (
        monthly["MoM_Revenue"].abs() > ANOMALY_THRESHOLD
    )

    # 标记 12 月为数据不完整（只有 9 天），单独处理
    monthly["IsPartial"] = monthly["Month"].astype(str) == "2011-12"

    return monthly


# ── 图11：月度环比变化率（标注异动点）────────────────────────────────────────
def plot_mom_change(monthly: pd.DataFrame) -> None:
    # 排除第一个月（无环比）和不完整的最后一个月
    data = monthly[~monthly["IsPartial"] & monthly["MoM_Revenue"].notna()].copy()
    data["Month_dt"] = data["Month"].dt.to_timestamp()

    colors = [ALERT if v else PRIMARY for v in data["IsAnomaly"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # 上图：收入绝对值
    ax1.fill_between(data["Month_dt"], data["Revenue"],
                     alpha=0.15, color=PRIMARY)
    ax1.plot(data["Month_dt"], data["Revenue"],
             color=PRIMARY, linewidth=2, marker="o", markersize=4)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax1.set_title("Monthly Revenue & MoM Change Rate")
    ax1.set_ylabel("Revenue")

    # 下图：环比变化率柱状图
    ax2.bar(data["Month_dt"], data["MoM_Revenue"] * 100,
            color=colors, width=20, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline( ANOMALY_THRESHOLD * 100, color=ALERT,
                linewidth=1, linestyle="--", alpha=0.5)
    ax2.axhline(-ANOMALY_THRESHOLD * 100, color=ALERT,
                linewidth=1, linestyle="--", alpha=0.5)

    # 标注异动月份的变化率数值
    for _, row in data[data["IsAnomaly"]].iterrows():
        sign = "+" if row["MoM_Revenue"] > 0 else ""
        ax2.annotate(
            f'{sign}{row["MoM_Revenue"]*100:.0f}%',
            xy=(row["Month_dt"], row["MoM_Revenue"] * 100),
            xytext=(0, 8 if row["MoM_Revenue"] > 0 else -14),
            textcoords="offset points",
            ha="center", fontsize=8, color=ALERT, fontweight="bold"
        )

    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax2.set_ylabel("MoM Change")
    ax2.set_xlabel("Month")

    plt.xticks(rotation=30)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "11_mom_revenue_change.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图12：异动月份驱动因素拆解 ───────────────────────────────────────────────
def plot_anomaly_breakdown(monthly: pd.DataFrame) -> None:
    """
    对每个异动月份，用三个子指标的环比变化来拆解原因：
    Revenue 环比 = AvgOrderValue 环比 × Orders 环比
    如果 Revenue 涨，是因为客单价涨了，还是订单量涨了，还是两者都涨？
    """
    data = monthly[
        monthly["IsAnomaly"] & ~monthly["IsPartial"]
    ].copy()

    if len(data) == 0:
        print("  [跳过] 没有达到阈值的异动月份")
        return

    # 计算三个维度的环比
    data["MoM_Orders"]    = monthly["Orders"].pct_change()[data.index]
    data["MoM_Customers"] = monthly["Customers"].pct_change()[data.index]
    data["MoM_AOV"]       = monthly["AvgOrderValue"].pct_change()[data.index]

    months = data["Month"].astype(str).tolist()
    n = len(months)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    metrics      = ["MoM_Revenue", "MoM_Orders", "MoM_Customers", "MoM_AOV"]
    metric_names = ["收入\nRevenue", "订单数\nOrders",
                    "活跃客户\nCustomers", "客单价\nAvg Order"]

    for ax, (_, row) in zip(axes, data.iterrows()):
        vals   = [row[m] * 100 for m in metrics]
        colors = [ALERT if v > 0 else PRIMARY for v in vals]
        bars   = ax.barh(metric_names, vals, color=colors, height=0.5)

        for bar, val in zip(bars, vals):
            sign = "+" if val > 0 else ""
            ax.text(val + (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:.1f}%",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=9, fontweight="bold")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"异动月份：{row['Month']}\n"
                     f"收入环比 {row['MoM_Revenue']*100:+.1f}%",
                     fontsize=10)
        ax.set_xlabel("MoM Change (%)")

    plt.suptitle("Anomaly Breakdown — What Drove the Change?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "12_anomaly_breakdown.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图13：周度收入趋势 ────────────────────────────────────────────────────────
def plot_weekly_revenue(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """
    月度趋势可能掩盖月内的短期波动。周度图能捕捉更细粒度的异动，
    比如某周因节假日导致的骤升或骤降。
    """
    weekly = (
        df.groupby("Week")["TotalPrice"]
        .sum()
        .reset_index()
    )
    weekly["Week_dt"] = weekly["Week"].dt.to_timestamp()

    # 用滚动均值 ± 1.5 倍标准差标记异常周
    rolling_mean = weekly["TotalPrice"].rolling(4, center=True).mean()
    rolling_std  = weekly["TotalPrice"].rolling(4, center=True).std()
    upper = rolling_mean + 1.5 * rolling_std
    lower = rolling_mean - 1.5 * rolling_std
    is_spike = weekly["TotalPrice"] > upper

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(weekly["Week_dt"], lower, upper,
                    alpha=0.12, color=PRIMARY, label="正常区间 (±1.5σ)")
    ax.plot(weekly["Week_dt"], weekly["TotalPrice"],
            color=PRIMARY, linewidth=1.2, alpha=0.8, label="周收入")
    ax.plot(weekly["Week_dt"][is_spike], weekly["TotalPrice"][is_spike],
            "o", color=ALERT, markersize=6, label="异常高峰周", zorder=5)
    ax.plot(weekly["Week_dt"], rolling_mean,
            color="gray", linewidth=1.5, linestyle="--",
            alpha=0.6, label="4周滚动均值")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax.set_title("Weekly Revenue with Anomaly Detection (±1.5σ Band)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "13_weekly_revenue.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 控制台：异动月份详细报告 ──────────────────────────────────────────────────
def print_anomaly_report(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    anomalies = monthly[monthly["IsAnomaly"] & ~monthly["IsPartial"]]
    if len(anomalies) == 0:
        print("没有检测到异动月份。")
        return

    print("\n── 异动月份详细报告 ─────────────────────────────────────────────────")
    for _, row in anomalies.iterrows():
        direction = "⬆ 正向异动" if row["MoM_Revenue"] > 0 else "⬇ 负向异动"
        print(f"\n{direction}：{row['Month']}  收入环比 {row['MoM_Revenue']*100:+.1f}%")
        print(f"  收入:   £{row['Revenue']:>10,.0f}    订单数: {row['Orders']:>5,}")
        print(f"  客户数: {row['Customers']:>5,}            客单价: £{row['AvgOrderValue']:>8,.2f}")

        # 找出该月 Top 5 商品
        month_data = df[df["Month"] == row["Month"]]
        top5 = (
            month_data.groupby("Description")["TotalPrice"]
            .sum().nlargest(5)
        )
        print("  Top 5 商品（按收入）:")
        for desc, rev in top5.items():
            short = desc[:45] + "…" if len(desc) > 45 else desc
            print(f"    £{rev:>8,.0f}  {short}")

        # 找出该月 Top 3 国家
        top3_country = (
            month_data.groupby("Country")["TotalPrice"]
            .sum().nlargest(3)
        )
        print("  Top 3 市场:")
        for country, rev in top3_country.items():
            print(f"    £{rev:>8,.0f}  {country}")

    print("\n── 业务解读 ────────────────────────────────────────────────────────")
    print("  2011-11 正向异动：圣诞季提前备货，客户大量采购礼品，是正常季节性峰值。")
    print("  其他负向异动月份：需结合当月新客获取情况和老客活跃度综合判断。")
    print("  注：2011-12 数据仅覆盖 9 天，收入骤降为数据截断所致，非真实异动。")
    print("─────────────────────────────────────────────────────────────────────")


def main():
    df      = load_data(DATA_PATH)
    monthly = build_monthly_metrics(df)

    print(f"\n[异动检测] 阈值: 月度环比变化 > ±{ANOMALY_THRESHOLD*100:.0f}%")
    anomaly_months = monthly[monthly["IsAnomaly"] & ~monthly["IsPartial"]]["Month"].tolist()
    print(f"  检测到 {len(anomaly_months)} 个异动月份: "
          + ", ".join(str(m) for m in anomaly_months))

    print("\n[绘图] 开始生成图表...")
    plot_mom_change(monthly)
    plot_anomaly_breakdown(monthly)
    plot_weekly_revenue(df, monthly)

    print_anomaly_report(df, monthly)
    print("\n✅ 异动分析完成")


if __name__ == "__main__":
    main()
