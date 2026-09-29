"""
forgetting-lab / summarize.py

eval_all.py가 남긴 런별 결과(영어 lm-eval JSON + 한국어 ko.csv)를 한 표로 합치고
그래프를 그린다. 런을 여러 개 주면 서로 비교한다.

  <root>/summary.csv                런 x 스텝 전체 지표
  <root>/forgetting_curve_<run>.png 런별 스텝 곡선 (왼쪽 영어, 오른쪽 한국어)
  <root>/tradeoff.png               핵심 그래프: 한국어 ppl vs LAMBADA ppl
  <root>/en_domains.png             영어 영역별(위키/책/웹) ppl의 원본 대비 % 변화
  <root>/final_comparison.csv       런별 마지막 스텝의 모든 지표, 원본 대비 변화

예)
  python summarize.py                                        # 있는 런 전부
  python summarize.py --runs run01_ko_wiki run02_warmup100
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

# ===== 설정 =====
DEFAULT_ROOT = "results/fp16"

# 뽑아올 지표: (컬럼 이름, lm-eval 태스크, 지표 키)
METRICS = [
    ("arc_easy_acc", "arc_easy", "acc,none"),
    ("hellaswag_acc_norm", "hellaswag", "acc_norm,none"),
    ("piqa_acc_norm", "piqa", "acc_norm,none"),
    ("lambada_acc", "lambada_openai", "acc,none"),
    ("lambada_ppl", "lambada_openai", "perplexity,none"),
]
ACC_COLS = ["arc_easy_acc", "hellaswag_acc_norm", "piqa_acc_norm", "lambada_acc"]
LABEL_STEPS = {25, 50, 100, 200, 300}   # tradeoff 그래프에 번호를 붙일 스텝
EN_DOMAINS = ["wiki", "books", "web"]
EN_COLS = [f"en_{d}_ppl" for d in EN_DOMAINS]
EN_LABELS = {"en_wiki_ppl": "위키 (wikipedia)", "en_books_ppl": "책 (PG-19)",
             "en_web_ppl": "웹 (C4)"}
# 최종 비교 표에 넣을 지표. 정확도는 %p 차이, perplexity는 % 변화로 본다
FINAL_METRICS = ACC_COLS + ["lambada_ppl"] + EN_COLS + ["ko_ppl"]
ACC_LABELS = {
    "arc_easy_acc": "ARC-Easy (acc)",
    "hellaswag_acc_norm": "HellaSwag (acc_norm)",
    "piqa_acc_norm": "PIQA (acc_norm)",
    "lambada_acc": "LAMBADA (acc)",
}

# dataviz 팔레트: 카테고리 슬롯 1~4 (light) + 차트 잉크
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# ===============


def parse_args():
    p = argparse.ArgumentParser(description="런별 결과를 합치고 그래프를 그린다")
    p.add_argument("--runs", nargs="+", help="비교할 런 이름 (기본: 결과 폴더에 있는 전부)")
    p.add_argument("--root", default=DEFAULT_ROOT, help="eval_all.py의 --results_root")
    return p.parse_args()


# ---------- 읽기 ----------

def find_json(step_dir):
    # eval_all.py가 남긴 지문 파일이 있으면 거기 적힌 결과(= 지금 체크포인트의 결과)를 쓴다
    stamp = os.path.join(step_dir, "ckpt_fingerprint.json")
    if os.path.exists(stamp):
        with open(stamp, encoding="utf-8") as f:
            path = os.path.join(step_dir, json.load(f)["result_json"])
        if os.path.exists(path):
            return path
    hits = glob.glob(os.path.join(step_dir, "**", "results_*.json"), recursive=True)
    return sorted(hits)[-1] if hits else None


def read_metrics(path):
    with open(path, encoding="utf-8") as f:
        results = json.load(f)["results"]
    return {col: results.get(task, {}).get(key) for col, task, key in METRICS}


def read_ko(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["step", "ko_loss", "ko_ppl", "dtype"])
    ko = pd.read_csv(csv_path)
    return ko[["step", "ko_loss", "ko_ppl", "dtype"]]


def read_en(csv_path):
    """en_domains.csv (step, domain 세로형)를 step당 한 줄(en_wiki_ppl, ...)로 편다."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["step"] + EN_COLS)
    en = pd.read_csv(csv_path)
    wide = en.pivot_table(index="step", columns="domain", values="ppl", aggfunc="last")
    wide = wide.reindex(columns=EN_DOMAINS)
    wide.columns = [f"en_{d}_ppl" for d in wide.columns]
    return wide.reset_index()


def discover_runs(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if d != "base" and os.path.isdir(os.path.join(root, d)))


def load_run(root, run):
    """한 런의 영어 + 한국어 결과를 step 기준으로 합친다 (step 0 = 공유 baseline)."""
    base_dir = os.path.join(root, "base")
    step_dirs = [(0, os.path.join(base_dir, "step-0"))]
    for d in sorted(glob.glob(os.path.join(root, run, "ckpt-*"))):
        step_dirs.append((int(os.path.basename(d).split("-")[1]), d))

    eng = []
    for step, d in sorted(step_dirs):
        path = find_json(d)
        if path is None:
            print(f"  [건너뜀] {run} step {step}: lm-eval JSON 없음 ({d})")
            continue
        eng.append({"step": step, **read_metrics(path)})
    if not eng:
        return pd.DataFrame()

    ko = pd.concat([read_ko(os.path.join(base_dir, "ko.csv")),
                    read_ko(os.path.join(root, run, "ko.csv"))], ignore_index=True)
    ko = ko.drop_duplicates(subset="step", keep="last")

    en = pd.concat([read_en(os.path.join(base_dir, "en_domains.csv")),
                    read_en(os.path.join(root, run, "en_domains.csv"))], ignore_index=True)
    en = en.drop_duplicates(subset="step", keep="last")

    df = (pd.DataFrame(eng).merge(ko, on="step", how="left")
          .merge(en, on="step", how="left").sort_values("step"))
    df.insert(0, "run", run)
    df.insert(1, "model", ["base" if s == 0 else f"ckpt-{s}" for s in df["step"]])
    return df.reset_index(drop=True)


# ---------- 그리기 ----------

def use_korean_font():
    """한글 라벨이 네모로 깨지지 않게 시스템 한글 폰트를 잡는다."""
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Malgun Gothic", "NanumGothic", "Noto Sans KR", "Gulim"]:
        if cand in have:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def style(ax, title, xlabel, ylabel, pad=12):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=pad)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def spread(ys, min_gap):
    """직접 라벨이 겹치지 않게 y 위치만 살짝 벌린다 (원래 순서 유지)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k, i in enumerate(order):
        if k and out[i] - out[order[k - 1]] < min_gap:
            out[i] = out[order[k - 1]] + min_gap
    return out


def set_step_ticks(ax, xs):
    if len(xs) <= 8:
        ax.set_xticks(xs)


STEP_XLABEL = "학습 스텝 (step 0 = 원본 pythia-410m)"


def plot_run_curve(df, run, path):
    """런 하나의 스텝 곡선: 왼쪽 영어 벤치마크, 오른쪽 한국어 perplexity."""
    xs = df["step"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.patch.set_facecolor(PAGE)

    # --- 왼쪽: 영어 벤치마크 (전부 정확도라 축 하나로 충분) ---
    style(ax1, "영어 벤치마크 정확도", STEP_XLABEL, "accuracy", pad=30)
    ax1.grid(axis="x", visible=False)
    ends = []
    for col, color in zip(ACC_COLS, SERIES):
        ys = df[col].tolist()
        ax1.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, label=ACC_LABELS[col],
                 clip_on=False, zorder=3)
        ends.append(ys[-1])

    # 색만으로 구분하지 않도록 선 끝에 이름을 직접 붙인다
    span = max(ends) - min(ends)
    for (col, color), y in zip(zip(ACC_COLS, SERIES), spread(ends, max(span, 0.2) * 0.13)):
        ax1.annotate(ACC_LABELS[col], xy=(xs[-1] + max(xs) * 0.03, y), color=color,
                     fontsize=9, fontweight="bold", va="center", annotation_clip=False)
    ax1.set_xlim(-max(xs) * 0.03, xs[-1] + max(xs) * 0.5)
    set_step_ticks(ax1, xs)
    # 레전드는 선 위에 겹치지 않게 그림 영역 바깥 한 줄로
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=1.6, columnspacing=1.3,
               handletextpad=0.5, borderpad=0)

    # --- 오른쪽: 한국어 held-out perplexity ---
    style(ax2, "한국어 held-out perplexity (낮을수록 잘함)", STEP_XLABEL, "perplexity", pad=30)
    ax2.grid(axis="x", visible=False)
    ys = df["ko_ppl"].tolist()
    ax2.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.2, clip_on=False, zorder=3)
    lo, hi = min(ys), max(ys)
    for x, y in [(xs[0], ys[0]), (xs[-1], ys[-1])]:
        ax2.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 11), textcoords="offset points",
                     color=SERIES[0], fontsize=10, fontweight="bold", ha="center")
    ax2.set_xlim(-max(xs) * 0.03, xs[-1] + max(xs) * 0.03)
    set_step_ticks(ax2, xs)
    ax2.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.28)

    fig.suptitle(f"{run} · 한국어를 배우는 동안 영어를 얼마나 잊는가",
                 color=INK, fontsize=15, fontweight="bold", x=0.035, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=PAGE)
    plt.close(fig)


def plot_tradeoff(dfs, path):
    """핵심 그래프: 한국어를 이만큼 얻을 때 영어를 얼마나 잃었나. 왼쪽 아래일수록 좋다."""
    fig, ax = plt.subplots(figsize=(9, 6.6))
    fig.patch.set_facecolor(PAGE)
    legend_ncol = 4 if len(dfs) <= 3 else 3
    legend_rows = -(-(len(dfs) + 1) // legend_ncol)   # 런들 + base 마커
    style(ax, "같은 한국어 실력을 얻을 때 영어를 얼마나 잃는가",
          "한국어 held-out perplexity  (낮을수록 한국어를 잘함)",
          "LAMBADA perplexity  (낮을수록 영어를 잘함)",
          pad=44 + 20 * (legend_rows - 1))
    # 안내문은 데이터와 겹칠 일 없게 제목 밑, 범례(줄 수만큼) 위에 붙인다
    ax.annotate("점 = 체크포인트(숫자는 스텝) · 왼쪽 아래로 갈수록 두 언어 모두 좋음",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 8 + 20 * legend_rows),
                textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.margins(0.13)

    labels, points = [], []
    families = []   # 같은 설정의 시드 반복(run01_ko_wiki, run01_seed43 …)은 같은 색, 선 모양만 다르게
    styles = ["-", "--", ":", "-."]
    seen = {}
    for run, df in dfs.items():
        fam = run.split("_")[0]
        if fam not in families:
            families.append(fam)
        k = seen.get(fam, 0)
        seen[fam] = k + 1
        d = df.dropna(subset=["ko_ppl", "lambada_ppl"])
        color = SERIES[families.index(fam) % len(SERIES)]
        ax.plot(d["ko_ppl"], d["lambada_ppl"], color=color, linewidth=2 if k == 0 else 1.6,
                linestyle=styles[k % len(styles)], marker="o" if k == 0 else "s",
                markersize=6 if k == 0 else 5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=run, zorder=3)

        # 스텝 번호는 몇 개만 (+ 런의 마지막 스텝). step 0은 아래 마커로 표시
        steps = d["step"].tolist()
        for x, y, s in zip(d["ko_ppl"], d["lambada_ppl"], steps):
            points.append((x, y, 3.5))
            if s in LABEL_STEPS or (s and s == steps[-1]):
                labels.append((x, y, f"{s}", color))

    # 모든 런이 공유하는 출발점
    base = next(iter(dfs.values())).iloc[0]
    ax.scatter([base["ko_ppl"]], [base["lambada_ppl"]], s=110, marker="D",
               color=INK, zorder=5, label="base (step 0)")
    points.append((base["ko_ppl"], base["lambada_ppl"], 7))

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=legend_ncol,
              frameon=False, fontsize=10, labelcolor=INK2, handlelength=2.4,
              columnspacing=1.6, handletextpad=0.6, borderpad=0)
    fig.tight_layout()
    place_labels(ax, labels, points)   # 레이아웃이 확정된 뒤에 자리를 잡아야 함
    fig.savefig(path, dpi=200, facecolor=PAGE)
    plt.close(fig)


def plot_en_domains(dfs, path):
    """런마다 패널 하나: 영어 세 영역 ppl이 원본 대비 몇 % 올랐나 (y축 공유)."""
    runs = [r for r, df in dfs.items() if df[EN_COLS].notna().any().any()]
    ncols = min(3, len(runs))
    nrows = -(-len(runs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + 0.8, 5.0 * nrows + 0.6),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(PAGE)
    xmax = max(dfs[r]["step"].max() for r in runs)
    flat = axes.flatten()
    for ax in flat[len(runs):]:
        ax.set_visible(False)

    for ax, run in zip(flat, runs):
        df = dfs[run]
        base = df[df["step"] == 0].iloc[0]
        style(ax, run, "학습 스텝", "원본 대비 perplexity 변화 (%)" if ax in axes[:, 0] else "",
              pad=10)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=INK2, linewidth=1, zorder=2)
        ends = []
        for col, color in zip(EN_COLS, SERIES):
            d = df.dropna(subset=[col])
            pct = (d[col] / base[col] - 1) * 100
            ax.plot(d["step"], pct, color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=EN_LABELS[col],
                    zorder=3)
            ends.append((d["step"].iloc[-1], pct.iloc[-1], col, color))
        # 선 끝에 최종 % 를 직접 적는다 (색만으로 구분하지 않게 영역 이름도)
        label_ys = spread([e[1] for e in ends], 7)   # 라벨끼리 안 겹치게 y만 살짝 벌림
        for (x, pct_end, col, color), y in zip(ends, label_ys):
            ax.annotate(f"{pct_end:+.0f}% {EN_LABELS[col].split(' ')[0]}",
                        xy=(x + xmax * 0.03, y), color=color, fontsize=9,
                        fontweight="bold", va="center", annotation_clip=False)
        ax.set_xlim(-xmax * 0.03, xmax * 1.32)

    # 제목·범례 자리는 인치로 고정해서 줄 수가 달라도 간격이 같게
    fig_h = fig.get_size_inches()[1]
    fig.suptitle("영어를 영역별로 얼마나 잊었나 · 0% = 원본 pythia-410m, 높을수록 많이 잊음",
                 color=INK, fontsize=14, fontweight="bold", x=0.02, ha="left",
                 y=1 - 0.14 / fig_h, va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, 1 - 0.52 / fig_h),
               ncol=3, frameon=False, fontsize=10, labelcolor=INK2, handlelength=1.8,
               columnspacing=1.6, borderpad=0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.56 / fig_h))
    fig.savefig(path, dpi=200, facecolor=PAGE)
    plt.close(fig)


def final_comparison(dfs):
    """행 = 지표, 열 = 원본 값과 런별 (마지막 스텝 값, 변화).
    정확도는 %p 차이, perplexity는 % 변화."""
    rows = []
    for m in FINAL_METRICS:
        is_acc = m in ACC_COLS
        base_val = next(iter(dfs.values())).query("step == 0")[m].iloc[0]
        row = {"metric": m, "better": "higher" if is_acc else "lower",
               "change_unit": "%p" if is_acc else "%", "base": base_val}
        for run, df in dfs.items():
            last = df.iloc[-1]
            v = last[m]
            col = f"{run}@{int(last['step'])}"
            row[col] = v
            if pd.isna(v) or pd.isna(base_val):
                row[f"{col}_change"] = float("nan")
            elif is_acc:
                row[f"{col}_change"] = (v - base_val) * 100
            else:
                row[f"{col}_change"] = (v / base_val - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def print_final(table):
    """콘솔용: 칸마다 '값 (변화)'로 합쳐서 보여준다."""
    runs = [c for c in table.columns if "@" in c and not c.endswith("_change")]
    out = pd.DataFrame({"지표": table["metric"],
                        "원본": ["-" if pd.isna(v) else f"{v:.4f}" if m in ACC_COLS
                                else f"{v:.3f}"
                                for m, v in zip(table["metric"], table["base"])]})
    for r in runs:
        cells = []
        for m, v, c, u in zip(table["metric"], table[r], table[f"{r}_change"],
                              table["change_unit"]):
            if pd.isna(v):
                cells.append("-")
            else:
                val = f"{v:.4f}" if m in ACC_COLS else f"{v:.3f}"
                cells.append(f"{val} ({c:+.1f}{u})")
        out[r] = cells
    print(out.to_string(index=False))
    print("  (정확도: 원본 대비 %p 차이, 높을수록 좋음 / perplexity: 원본 대비 % 변화, 낮을수록 좋음)")


# 라벨 후보 위치 (점 기준 오프셋 pt, ha, va). 앞쪽일수록 우선
LABEL_SPOTS = [(0, 8, "center", "bottom"), (0, -8, "center", "top"),
               (8, 0, "left", "center"), (-8, 0, "right", "center"),
               (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"),
               (7, -7, "left", "top"), (-7, -7, "right", "top"),
               (0, 19, "center", "bottom"), (0, -19, "center", "top"),
               (18, 0, "left", "center"), (-18, 0, "right", "center")]


def place_labels(ax, labels, points, fontsize=8.5):
    """점이 몰린 곳에서 스텝 번호끼리, 또는 번호와 점이 겹치지 않게 자리를 고른다.
    labels: (x, y, 글자, 색), points: (x, y, 반지름pt) — 피해야 할 점들."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = lambda x, y: ax.transData.transform((x, y)) * 72 / fig.dpi
    obstacles = [(*to_pt(x, y), r) for x, y, r in points]
    taken = []   # 이미 놓은 라벨 상자 (x0, y0, x1, y1), pt 단위

    def box(px, py, text, dx, dy, ha, va):
        w, h = len(text) * fontsize * 0.62, fontsize * 0.95
        x0 = px + dx - {"center": w / 2, "left": 0, "right": w}[ha]
        y0 = py + dy - {"center": h / 2, "bottom": 0, "top": h}[va]
        return x0, y0, x0 + w, y0 + h

    def free(b):
        pad = 1.0
        for t in taken:
            if b[0] < t[2] + pad and t[0] < b[2] + pad and b[1] < t[3] + pad and t[1] < b[3] + pad:
                return False
        for ox, oy, r in obstacles:
            if b[0] - r < ox < b[2] + r and b[1] - r < oy < b[3] + r:
                return False
        return True

    for x, y, text, color in labels:
        px, py = to_pt(x, y)
        spot = next((s for s in LABEL_SPOTS if free(box(px, py, text, *s))), LABEL_SPOTS[0])
        taken.append(box(px, py, text, *spot))
        dx, dy, ha, va = spot
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    color=color, fontsize=fontsize, fontweight="bold",
                    ha=ha, va=va, zorder=4)


def main():
    args = parse_args()
    runs = args.runs or discover_runs(args.root)
    if not runs:
        raise SystemExit(f"{args.root} 에서 런을 찾지 못했습니다. eval_all.py를 먼저 돌리세요.")

    print(f"런 {len(runs)}개 읽는 중: {', '.join(runs)}")
    dfs = {}
    for run in runs:
        df = load_run(args.root, run)
        if df.empty:
            print(f"  [건너뜀] {run}: 결과가 없습니다")
            continue
        dfs[run] = df
    if not dfs:
        raise SystemExit("읽을 수 있는 결과가 없습니다.")

    combined = pd.concat(dfs.values(), ignore_index=True)
    combined = combined[[c for c in combined.columns if c != "dtype"] + ["dtype"]]
    os.makedirs(args.root, exist_ok=True)
    out_csv = os.path.join(args.root, "summary.csv")
    combined.to_csv(out_csv, index=False)

    print("\n=== 런 x 스텝 요약 ===")
    print(combined.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    final = final_comparison(dfs)
    final_csv = os.path.join(args.root, "final_comparison.csv")
    final.to_csv(final_csv, index=False)
    print("\n=== 최종 비교: 런별 마지막 스텝 vs 원본 ===")
    print_final(final)

    if not use_korean_font():
        print("\n(한글 폰트를 못 찾아 그래프 글자가 깨질 수 있습니다)")

    made = [out_csv, final_csv]
    for run, df in dfs.items():
        p = os.path.join(args.root, f"forgetting_curve_{run}.png")
        plot_run_curve(df, run, p)
        made.append(p)
    p = os.path.join(args.root, "tradeoff.png")
    plot_tradeoff(dfs, p)
    made.append(p)
    if any(df[EN_COLS].notna().any().any() for df in dfs.values()):
        p = os.path.join(args.root, "en_domains.png")
        plot_en_domains(dfs, p)
        made.append(p)
    else:
        print("\n(영어 영역별 결과가 아직 없어 en_domains.png는 건너뜀: "
              "eval_all.py --only en_domains 를 먼저 돌리세요)")

    print("\n저장:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
