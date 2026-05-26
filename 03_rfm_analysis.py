"""
03_rfm_analysis.py
==================
RFM 客户分层分析

业务背景：
    并非所有客户的价值都相同。RFM 模型通过三个维度量化客户价值：
      - Recency   (R)：客户最近一次购买距今多少天？越近越好。
      - Frequency (F)：客户在观察窗口内下了多少笔订单？越多越好。
      - Monetary  (M)：客户累计消费了多少钱？越多越好。

    通过对 R/F/M 打分并组合，可以将客户分成不同群体，为差异化营销提供依据。

分析方法：
    1. 计算每位客户的 R/F/M 原始值
    2. 用五分位法（1-5 分）分别对 R/F/M 打分
    3. 根据 RFM 组合规则，给每位客户打上业务标签
    4. 输出各分群画像与业务建议

输出：
    data/rfm_result.csv            — 含分层标签的客户 RFM 表
    outputs/charts/06_rfm_segments.png — 分群客户数量分布
    outputs/charts/07_rfm_scatter.png  — R vs M 散点图（颜色区分群体）
    outputs/charts/08_rfm_heatmap.png  — RFM 分值热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "orders_for_rfm.csv")
OUT_CSV   = os.path.join(BASE_DIR, "data", "rfm_result.csv")
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


# ── RFM 分群规则 ──────────────────────────────────────────────────────────────
# 规则：(标签, 对 R_Score 的条件, 对 FM_Score 的条件)
# FM_Score = round((F_Score + M_Score) / 2, 0)
SEGMENT_RULES = [
    ("Champions",          lambda r, fm: (r >= 4) & (fm >= 4)),
    ("Loyal Customers",    lambda r, fm: (r >= 2) & (fm >= 3)),
    ("Potential Loyalists",lambda r, fm: (r >= 3) & (fm <= 3)),
    ("Recent Customers",   lambda r, fm: (r >= 4) & (fm <= 2)),
    ("Promising",          lambda r, fm: (r == 3) & (fm <= 2)),
    ("Need Attention",     lambda r, fm: (r == 2) & (fm == 2)),
    ("About to Sleep",     lambda r, fm: (r <= 2) & (fm <= 2)),
    ("At Risk",            lambda r, fm: (r <= 2) & (fm >= 3)),
    ("Can't Lose Them",    lambda r, fm: (r == 1) & (fm >= 4)),
    ("Hibernating",        lambda r, fm: (r == 1) & (fm <= 2)),
]

# 各分群对应的营销建议（用于报告）
STRATEGIES = {
    "Champions":           "奖励忠诚度计划，鼓励推荐好友，优先试用新品",
    "Loyal Customers":     "会员升级激励，交叉销售高毛利商品",
    "Potential Loyalists": "会员计划入门，发送产品教育内容提升频次",
    "Recent Customers":    "新手引导邮件，提供首次复购优惠券",
    "Promising":           "建立品牌认知，免费试用或入门礼包",
    "Need Attention":      "基于历史偏好定向推送，唤醒活跃度",
    "About to Sleep":      "发送限时优惠，询问产品体验反馈",
    "At Risk":             "Win-back 邮件序列，突出情感连接与专属折扣",
    "Can't Lose Them":     "电话/人工客服主动触达，VIP 专属福利",
    "Hibernating":         "低成本再激活广告，或从名单中移除降低成本",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"[加载] {len(df):,} 条订单，{df['CustomerID'].nunique():,} 位客户")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """以数据集最后一天的次日作为参考日期（分析截止日）。"""
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"[RFM] 参考日期（Snapshot Date）: {reference_date.date()}")

    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency =("InvoiceNo",   "nunique"),
        Monetary  =("TotalPrice",  "sum"),
    ).reset_index()

    print(f"\n── RFM 原始值统计 ──")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1).to_string())
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    五分位打分：
    - R：Recency 越小（越近）越好 → 用 ascending=False 分位
    - F、M：越大越好 → 用 ascending=True 分位
    """
    rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5,
                              labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                              q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"],  q=5,
                              labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)

    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + \
                       rfm["F_Score"].astype(str) + \
                       rfm["M_Score"].astype(str)
    rfm["FM_Score"]  = ((rfm["F_Score"] + rfm["M_Score"]) / 2).round(0).astype(int)
    return rfm


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """按优先级依次匹配分群规则（先匹配先生效）。"""
    rfm["Segment"] = "Other"
    for label, rule in SEGMENT_RULES:
        mask = rule(rfm["R_Score"], rfm["FM_Score"]) & (rfm["Segment"] == "Other")
        rfm.loc[mask, "Segment"] = label
    return rfm


# ── 图6：分群客户数量 ─────────────────────────────────────────────────────────
def plot_segments(rfm: pd.DataFrame) -> None:
    seg_count = rfm["Segment"].value_counts().sort_values(ascending=True)
    COLORS = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(seg_count)))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(seg_count.index, seg_count.values,
                   color=COLORS, height=0.65)
    for bar, val in zip(bars, seg_count.values):
        pct = val / len(rfm) * 100
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=8)

    ax.set_title("Customer Segments (RFM)")
    ax.set_xlabel("Number of Customers")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "06_rfm_segments.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图7：Recency vs Monetary 散点图 ──────────────────────────────────────────
def plot_rfm_scatter(rfm: pd.DataFrame) -> None:
    segments = rfm["Segment"].unique()
    cmap = plt.cm.get_cmap("tab10", len(segments))
    color_map = {seg: mcolors.to_hex(cmap(i)) for i, seg in enumerate(segments)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for seg in segments:
        sub = rfm[rfm["Segment"] == seg]
        ax.scatter(sub["Recency"], sub["Monetary"],
                   label=seg, alpha=0.5, s=20,
                   color=color_map[seg])

    ax.set_title("Customer Distribution: Recency vs Monetary")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (£)")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, frameon=False)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "07_rfm_scatter.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


# ── 图8：各分群平均 RFM 分值热力图 ───────────────────────────────────────────
def plot_rfm_heatmap(rfm: pd.DataFrame) -> None:
    seg_scores = (
        rfm.groupby("Segment")[["R_Score", "F_Score", "M_Score"]]
        .mean()
        .round(2)
        .sort_values("R_Score", ascending=False)
    )
    seg_scores.columns = ["Recency\nScore", "Frequency\nScore", "Monetary\nScore"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(seg_scores.values, cmap="RdYlGn",
                   vmin=1, vmax=5, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Score (1–5)")

    ax.set_xticks(range(3))
    ax.set_xticklabels(seg_scores.columns, fontsize=9)
    ax.set_yticks(range(len(seg_scores)))
    ax.set_yticklabels(seg_scores.index, fontsize=8)

    # 在格子中标注数值
    for i in range(len(seg_scores)):
        for j in range(3):
            ax.text(j, i, f"{seg_scores.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Average RFM Scores by Segment")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "08_rfm_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ 已保存: {out}")


def print_segment_report(rfm: pd.DataFrame) -> None:
    print("\n── RFM 分群报告 ──────────────────────────────────────────────────────")
    summary = (
        rfm.groupby("Segment")
        .agg(
            客户数=("CustomerID", "count"),
            平均Recency=("Recency", "mean"),
            平均Frequency=("Frequency", "mean"),
            平均Monetary=("Monetary", "mean"),
        )
        .round(1)
        .sort_values("客户数", ascending=False)
    )
    print(summary.to_string())

    print("\n── 各分群营销建议 ─────────────────────────────────────────────────────")
    for seg, strategy in STRATEGIES.items():
        count = (rfm["Segment"] == seg).sum()
        if count > 0:
            print(f"  [{seg}] ({count:,} 人) → {strategy}")


def main():
    df  = load_data(DATA_PATH)
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)

    rfm.to_csv(OUT_CSV, index=False)
    print(f"\n✅ RFM 结果已保存至: {OUT_CSV}")

    print("\n[绘图] 开始生成图表...")
    plot_segments(rfm)
    plot_rfm_scatter(rfm)
    plot_rfm_heatmap(rfm)

    print_segment_report(rfm)
    print("\n✅ RFM 分析完成")


if __name__ == "__main__":
    main()
