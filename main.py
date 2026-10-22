import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

# 1) 데이터 로드 및 출력 폴더 생성
os.makedirs("./imgs", exist_ok=True)
df = pd.read_csv("train.csv")  # Kaggle에서 내려받은 train.csv 사용

# 2) 데이터 결측치 확인
print("shape:", df.shape)              # 데이터 크기 확인
print("null counts:\n", df.isnull().sum())  # 결측치 개수 확인

# 3) 전처리
df["Age"] = df["Age"].fillna(df["Age"].median())     # Age는 중앙값으로 대치
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Embarked는 최빈값으로 대치

# 4) 파생 변수 처리
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # 본인 포함 가족 수
df["FamilyGroup"] = np.select(
    [df["FamilySize"] == 1, df["FamilySize"] <= 4],
    ["Alone", "2-4"],
    default="5+",
)
df["HasFamily"] = np.where(df["FamilySize"] > 1, "Family", "Alone")
df["IsFemale"] = np.where(df["Sex"] == "female", 1, 0)  # 평균 = 여성 비율
df["IsFirst"] = np.where(df["Pclass"] == 1, 1, 0)       # 평균 = 1등석 비율

# 5) GroupBy
# 5-1) 가족 규모별 생존율·인원·여성 비율·1등석 비율 (인사이트 1)
group_order = ["Alone", "2-4", "5+"]
family_summary = (
    df.groupby("FamilyGroup")
    .agg(
        SurvivalRate=("Survived", "mean"),
        Count=("Survived", "size"),
        FemaleRatio=("IsFemale", "mean"),
        FirstClassRatio=("IsFirst", "mean"),
    )
    .reindex(group_order)
)
print("\nGroupBy - Summary by Family Group:\n", family_summary.round(3))

# 5-2) 가족 동반 여부별 생존율·여성 비율 (인사이트 2)
has_family = df.groupby("HasFamily").agg(
    SurvivalRate=("Survived", "mean"),
    FemaleRatio=("IsFemale", "mean"),
)
print("\nGroupBy - Alone vs Family:\n", has_family.round(3))

# 6) Pivot Table: 성별·가족 동반 × 좌석등급 생존율 및 인원 (인사이트 3)
pivot_family = pd.pivot_table(
    df, values="Survived", index=["Sex", "HasFamily"], columns="Pclass", aggfunc="mean"
)
pivot_count = pd.pivot_table(
    df, values="Survived", index=["Sex", "HasFamily"], columns="Pclass", aggfunc="size"
)
print("\nPivot Table - Survival rate by Sex/HasFamily and Pclass:\n", pivot_family.round(3))
print("\nPivot Table - Count:\n", pivot_count)

# 7) 시각화 (영문 라벨 권장)
# 7-1) 가족 규모별 생존율 (역U자형, 인사이트 1)
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(group_order))
rates = family_summary["SurvivalRate"].to_numpy()
overall = df["Survived"].mean()
ax.plot(x, rates, marker="o", markersize=10, linewidth=2.5, color="#2b6cb0")
ax.axhline(overall, color="gray", linestyle="--", linewidth=1, label=f"Overall ({overall:.1%})")
for xi, r, n in zip(x, rates, family_summary["Count"]):
    below = r < overall  # 평균선보다 낮은 점은 라벨을 아래에 배치해 점선과 겹치지 않게 함
    ax.annotate(f"{r:.1%}\n(n={n})", (xi, r), textcoords="offset points",
                xytext=(0, -14 if below else 12), ha="center", va="top" if below else "bottom")
ax.set_xticks(x, ["Alone (1)", "Small (2-4)", "Large (5+)"])
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 0.8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family Size (incl. self)")
ax.set_ylabel("Survival Rate")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("./imgs/family_size_survival.png", dpi=200)

# 7-2) 가족 동반 그룹의 생존율·여성 비율 비교 (인사이트 2) + 성별·가족 × 등급 피벗 히트맵 (인사이트 3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})

cats = ["Survival Rate", "Female Ratio"]
alone_vals = has_family.loc["Alone", ["SurvivalRate", "FemaleRatio"]].to_numpy()
family_vals = has_family.loc["Family", ["SurvivalRate", "FemaleRatio"]].to_numpy()
xb = np.arange(len(cats))
w = 0.38
bars_a = ax1.bar(xb - w / 2, alone_vals, w, color="#a0aec0", label="Alone")
bars_f = ax1.bar(xb + w / 2, family_vals, w, color="#dd6b20", label="With family")
ax1.bar_label(bars_a, labels=[f"{v:.1%}" for v in alone_vals], padding=3)
ax1.bar_label(bars_f, labels=[f"{v:.1%}" for v in family_vals], padding=3)
ax1.set_xticks(xb, cats)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylabel("Ratio")
ax1.set_title("Alone vs With Family: Survival Rate and Female Ratio")
ax1.legend(loc="upper right")

vals = pivot_family.to_numpy()
counts = pivot_count.to_numpy()
im = ax2.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(vals.shape[1]), ["1st", "2nd", "3rd"])
ax2.set_yticks(range(vals.shape[0]), [f"{s.capitalize()} / {h}" for s, h in pivot_family.index])
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax2.text(j, i, f"{vals[i, j]:.1%}\n(n={counts[i, j]})", ha="center", va="center", fontsize=10)
ax2.add_patch(Rectangle((1.5, -0.5), 1, 2, fill=False, edgecolor="black", linewidth=3))  # 3등석 여성 강조
ax2.set_xlabel("Pclass")
ax2.set_title("Survival Rate: Sex / Family x Pclass (3rd-class women boxed)")
fig.colorbar(im, ax=ax2, format=PercentFormatter(1.0))
fig.tight_layout()
fig.savefig("./imgs/family_sex_pclass_survival.png", dpi=200)

# 8) 인사이트 출력
f3_alone = pivot_family.loc[("female", "Alone"), 3]
f3_family = pivot_family.loc[("female", "Family"), 3]
insights = [
    f"가족 수와 생존율은 역U자형이다: 혼자 {rates[0]:.1%} → 2~4명 {rates[1]:.1%} → 5명 이상 {rates[2]:.1%}",
    f"가족 동반자 생존율({has_family.loc['Family', 'SurvivalRate']:.1%})이 혼자({has_family.loc['Alone', 'SurvivalRate']:.1%})보다 높아 보이는 것은 "
    f"가족 동반 그룹의 여성 비율({has_family.loc['Family', 'FemaleRatio']:.1%})이 혼자 그룹({has_family.loc['Alone', 'FemaleRatio']:.1%})보다 높기 때문이다",
    f"3등석 여성은 가족이 있으면 생존율이 {f3_alone:.1%} → {f3_family:.1%}로 낮아진다. 아이들과 함께 움직이느라 탈출이 어려웠을 가능성이 있다",
]
print("\nInsights:")
for i in insights:
    print("-", i)