"""
02_eda.py
=========
探索性数据分析（EDA）

业务目标：
    回答管理层最关心的几个基本问题——
    1. 我们的收入趋势如何？有无明显的季节性？
    2. 哪些商品贡献了最多收入？
    3. 哪些市场最重要？
    4. 用户在什么时间段最活跃？

输出（保存至 outputs/charts/）：
    01_monthly_revenue.png  — 月度收入趋势
    02_top_products.png     — Top 10 商品（按收入）
    03_country_revenue.png  — 各国收入分布（除英国）
    04_hourly_orders.png    — 订单时段分布
    05_dow_orders.png       — 星期几订单分布
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "orders_for_rfm.csv")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ── 全局绘图风格 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})
PRIMARY = "#2E86AB"
ACCENT  = "#E84855"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    print(f"[加载] {len(df):,} 条正常订单")
    return df


# ── 图1：月度收入趋势 ─────────────────────────────────────────────────────────
def plot_monthly_revenue(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby("Month")["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly["Month_dt"] = monthly["Month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(monthly["Month_dt"], monthly["TotalPrice"],
                    alpha=0.15, color=PRIMARY)
    ax.plot(monthly["Month_dt"], monthly["TotalPrice"],
            color=PRIMARY, linewidth=2, marker="o", markersize=4)

    # 标注最高点
    peak = monthly.loc[monthly["TotalPrice"].idxmax()]
    ax.annotate(
        f"Peak\n£{peak['TotalPrice']/1e6:.2f}M",
        xy=(peak["Month_dt"], peak["TotalPrice"]),
        xytext=(0, 15), textcoords="offset points",
        ha="center", fontsize=9, color=ACCENT,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2)
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M")
    )
    ax.set_title("Monthly Revenue (Dec 2010 – Dec 2011)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "01_monthly_revenue.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图2：Top 10 商品 ──────────────────────────────────────────────────────────
def plot_top_products(df: pd.DataFrame) -> None:
    top10 = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top10.index, top10.values, color=PRIMARY, height=0.6)

    # 在条形右侧标注金额
    for bar, val in zip(bars, top10.values):
        ax.text(val + 500, bar.get_y() + bar.get_height() / 2,
                f"£{val/1e3:.0f}K", va="center", fontsize=8)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K")
    )
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Total Revenue")
    # 截断过长的商品名
    ax.set_yticklabels(
        [name[:40] + "…" if len(name) > 40 else name for name in top10.index],
        fontsize=8
    )
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "02_top_products.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图3：各国收入（除英国）────────────────────────────────────────────────────
def plot_country_revenue(df: pd.DataFrame) -> None:
    country_rev = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")["TotalPrice"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(country_rev.index, country_rev.values,
                   color=ACCENT, height=0.6)
    for bar, val in zip(bars, country_rev.values):
        ax.text(val + 200, bar.get_y() + bar.get_height() / 2,
                f"£{val/1e3:.0f}K", va="center", fontsize=8)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K")
    )
    ax.set_title("Top 10 International Markets by Revenue\n(UK Excluded)")
    ax.set_xlabel("Total Revenue")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "03_country_revenue.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图4：每小时订单量 ─────────────────────────────────────────────────────────
def plot_hourly_orders(df: pd.DataFrame) -> None:
    hourly = (
        df.groupby("Hour")["InvoiceNo"]
        .nunique()
        .reindex(range(24), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly.index, hourly.values, color=PRIMARY, width=0.7)
    ax.set_title("Order Volume by Hour of Day")
    ax.set_xlabel("Hour (UTC+0)")
    ax.set_ylabel("Number of Orders")
    ax.set_xticks(range(24))
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "04_hourly_orders.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图5：星期几订单分布 ───────────────────────────────────────────────────────
def plot_dow_orders(df: pd.DataFrame) -> None:
    DOW_ORDER = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        df.groupby("DayOfWeek")["InvoiceNo"]
        .nunique()
        .reindex(DOW_ORDER, fill_value=0)
    )

    colors = [ACCENT if d == "Sunday" else PRIMARY for d in DOW_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow.index, dow.values, color=colors, width=0.6)
    ax.set_title("Order Volume by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Orders")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "05_dow_orders.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 控制台摘要统计 ────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    total_rev = df["TotalPrice"].sum()
    avg_order = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()
    uk_share  = (
        df[df["Country"] == "United Kingdom"]["TotalPrice"].sum() / total_rev
    )
    print("\n── 业务摘要 ──────────────────────────────────────────")
    print(f"  总收入:       £{total_rev:>12,.0f}")
    print(f"  平均订单金额: £{avg_order:>12,.2f}")
    print(f"  英国市场占比:  {uk_share*100:.1f}%")
    print(f"  活跃客户数:   {df['CustomerID'].nunique():>12,}")
    print(f"  SKU 数量:     {df['StockCode'].nunique():>12,}")
    print("──────────────────────────────────────────────────────")


def main():
    df = load_data(DATA_PATH)
    print_summary(df)

    print("\n[绘图] 开始生成图表...")
    plot_monthly_revenue(df)
    plot_top_products(df)
    plot_country_revenue(df)
    plot_hourly_orders(df)
    plot_dow_orders(df)
    print("\n✅ EDA 图表全部保存至 outputs/charts/")


if __name__ == "__main__":
    main()
