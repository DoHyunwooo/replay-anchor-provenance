"""
forgetting-lab / make_figures.py

논문용 그림 4개 (LAB_NOTES 9장, 워크숍 논문 초안 PAPER_OUTLINE.md). 결과 파일만 읽고 아무것도 다시 계산·학습하지 않는다.

  fig1_trunks   : B·C1 줄기 궤적. (a) 영어 ppl 원본 대비 % (세 영역 평균), (b) 한국어 ppl의 c1 대비 %.
  fig2_branches : C2·C4 분기 막대. (a) 영어 원본 대비 % (세 영역 평균), (b) 중간 지식 (ko/ja/ru 평균, M_3 대비 %).
  fig3_age      : C3 anchor 나이 곡선. (a) 영어, (b) 중간 지식. 혼합 50/50은 나이 축 밖의 좁은 패널에 따로 표시,
                  사후 관찰(base_full + 교사의 영어 저하)은 점선.
  fig4_p5       : P5 산점도. 영어 원본 대비 % vs probe KL(원본‖체크포인트), vs 파라미터 거리. rho는 c_check.json 값.

입력: results/cycles/base/eval.csv, results/cycles/trunk_*/eval.csv, results/cycles/branches/trunk_fresh_s4x/eval.csv,
      results/cycles/c_check.json
출력: results/figures/<이름>.png (300dpi), .pdf, 그리고 그림마다 데이터 표 <이름>_data.csv (표 보기·재현용)

색 (dataviz 기준 팔레트 1~3번, 검증 통과: CVD ΔE 9.2, 정상 시각 ΔE 24.0 전 쌍): 파랑 #2a78d6, 주황 #eb6834, 청록 #1baf7a.
청록은 밝은 바탕 대비 3:1 미만이라 직접 라벨·모양·데이터 표를 같이 둔다. 색만으로 구분하지 않는다 (모양·선 스타일·라벨).

예)
  python make_figures.py
"""
import csv
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_summary import read_eval

R = "results/cycles"
OUT = "results/figures"
EN = ["en_wiki", "en_books", "en_web"]
MID = ["ko", "ja", "ru"]
SEEDS = [42, 43]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#898781"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": SURFACE, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

base = read_eval(os.path.join(R, "base", "eval.csv"))["c0_base"]
trunk = {t: read_eval(os.path.join(R, t, "eval.csv")) for t in
         [f"trunk_{m}_s{s}" for m in ("FT", "fresh", "base") for s in SEEDS]}
branch = {s: read_eval(os.path.join(R, "branches", f"trunk_fresh_s{s}", "eval.csv")) for s in SEEDS}
check = json.load(open(os.path.join(R, "c_check.json"), encoding="utf-8"))


def pct(a, b):
    return (a / b - 1) * 100


def en_mean(ev):
    return st.mean(pct(ev[f"ppl_{d}"], base[f"ppl_{d}"]) for d in EN)


def mid_mean(ev, ref):
    return st.mean(pct(ev[f"ppl_{l}"], ref[f"ppl_{l}"]) for l in MID)


def save(fig, name, rows, header):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    with open(os.path.join(OUT, f"{name}_data.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    plt.close(fig)
    print(f"저장: {OUT}/{name}.png, .pdf, _data.csv ({len(rows)}행)")


def end_labels(ax, x, items, min_gap):
    """선 끝 직접 라벨. items = [(y, text)]. 가까운 라벨끼리 min_gap(데이터 단위) 이상 벌린다."""
    items = sorted(items)
    placed = []
    for y, text in items:
        y2 = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y2)
        ax.annotate(text, (x, y), xytext=(x + 0.12, y2), textcoords="data", va="center", fontsize=8, color=INK2)


# ---------------- Fig 1: 줄기 궤적 ----------------
def fig1():
    conds = [("FT", "FT (no replay, 120 steps)", AQUA, "^"), ("fresh", "fresh (latest anchor)", ORANGE, "o"),
             ("base", "base anchor", BLUE, "s")]
    tags = ["c0_base", "c1_ko", "c2_ja", "c3_ru"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    rows, lab_a, lab_b = [], [], []
    for key, label, color, mk in conds:
        ends_a, ends_b = [], []
        for s, ls in zip(SEEDS, ("-", "--")):
            ev = trunk[f"trunk_{key}_s{s}"]
            ya = [0.0] + [en_mean(ev[t]) for t in tags[1:]]
            yb = [0.0] + [pct(ev[t]["ppl_ko"], ev["c1_ko"]["ppl_ko"]) for t in tags[2:]]
            axes[0].plot(range(4), ya, ls=ls, color=color, lw=2, marker=mk, ms=5, mec=SURFACE, mew=1.2, zorder=3)
            axes[1].plot(range(1, 4), yb, ls=ls, color=color, lw=2, marker=mk, ms=5, mec=SURFACE, mew=1.2, zorder=3)
            ends_a.append(ya[-1]); ends_b.append(yb[-1])
            for t, a in zip(tags, ya):
                rows.append([key, s, t, round(a, 3), round(pct(ev[t]["ppl_ko"], ev["c1_ko"]["ppl_ko"]), 3) if t != "c0_base" else ""])
        lab_a.append((st.mean(ends_a), label.split(" (")[0]))
        lab_b.append((st.mean(ends_b), label.split(" (")[0]))
    end_labels(axes[0], 3, lab_a, 2.0)
    # (b)는 FT와 base anchor 선이 끝에서 교차해 끝 라벨이 어느 선인지 모호하다 → 라벨 없이 위 범례(색·모양)로 구분
    axes[0].set_xticks(range(4), ["base", "c1 (ko)", "c2 (ja)", "c3 (ru)"])
    axes[0].set_ylabel("English ppl change vs base (%)")
    axes[0].set_title("(a) English forgetting (wiki/books/web mean)", loc="left")
    axes[1].set_xticks(range(1, 4), ["c1 (ko)", "c2 (ja)", "c3 (ru)"])
    axes[1].set_ylabel("Korean ppl change vs c1 (%)")
    axes[1].set_title("(b) Korean retention after learning it", loc="left")
    axes[0].set_xlim(axes[0].get_xlim()[0], axes[0].get_xlim()[1] + 0.9)   # 끝 라벨 자리는 (a)에만
    handles = [plt.Line2D([], [], color=c, marker=m, lw=2, label=l) for _, l, c, m in conds]
    handles += [plt.Line2D([], [], color=INK2, lw=1.5, ls="-", label="seed 42"),
                plt.Line2D([], [], color=INK2, lw=1.5, ls="--", label="seed 43")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig1_trunks", rows, ["condition", "seed", "cycle", "english_mean_pct_vs_base", "korean_pct_vs_c1"])


# ---------------- Fig 2: C2·C4 분기 막대 ----------------
def fig2():
    arms = [("ft150", "FT-150", NEUTRAL), ("latest_full", "latest\nNTP", ORANGE), ("latest_ep", "latest\nNTP-EP", ORANGE),
            ("latest_kl", "latest\nKL", ORANGE), ("base_full", "base\nNTP", BLUE), ("base_ep", "base\nNTP-EP", BLUE),
            ("base_kl", "base\nKL", BLUE)]
    m3 = {s: trunk[f"trunk_fresh_s{s}"]["c3_ru"] for s in SEEDS}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    rows = []
    for ax, kind in zip(axes, ("en", "mid")):
        for i, (arm, label, color) in enumerate(arms):
            vals = [en_mean(branch[s][arm]) if kind == "en" else mid_mean(branch[s][arm], m3[s]) for s in SEEDS]
            mean = st.mean(vals)
            hatch = "////" if arm.endswith("_kl") else ("...." if arm.endswith("_ep") else None)
            ax.bar(i, mean, width=0.72, color=color, alpha=0.9 if hatch is None else 0.55, edgecolor=SURFACE,
                   linewidth=2, hatch=hatch, zorder=2)
            ax.scatter([i - 0.12, i + 0.12], vals, s=18, color=INK2, edgecolor=SURFACE, linewidth=1, zorder=4)
            for s, v in zip(SEEDS, vals):
                rows.append([kind, arm, s, round(v, 3)])
            # 막대 값 라벨은 달지 않는다 (일부만 달면 일관성이 없음, 수치는 표 2)
        ax.set_xticks(range(len(arms)), [a[1] for a in arms], fontsize=7.5)
        ax.axvline(0.5, color=GRID, lw=1)
        ax.axvline(3.5, color=GRID, lw=1)
    m3_en = st.mean(en_mean(m3[s]) for s in SEEDS)
    axes[0].axhline(m3_en, color=INK2, ls="--", lw=1.2, zorder=3)
    axes[0].annotate(f"M_3 (branch start)\n{m3_en:.1f}", (6.45, m3_en), xytext=(0, 4), textcoords="offset points",
                     ha="right", va="bottom", fontsize=7.5, color=INK2)
    for s in SEEDS:
        rows.append(["en", "M3", s, round(en_mean(m3[s]), 3)])
    axes[0].set_ylabel("English ppl change vs base (%)")
    axes[0].set_title("(a) Original knowledge: English (lower is better)", loc="left")
    axes[1].set_ylabel("ko/ja/ru ppl change vs M_3 (%)")
    axes[1].set_title("(b) Intermediate knowledge (lower is better)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="latest anchor (M_3)"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, label="base anchor"),
               plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, label="no replay (FT-150)"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="....", label="English-prose filter"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="////", label="KL objective (C4)"),
               plt.Line2D([], [], marker="o", color=INK2, lw=0, ms=4, label="per trunk (2)")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_branches", rows, ["panel", "arm", "trunk_seed", "value_pct"])


# ---------------- Fig 3: C3 나이 곡선 ----------------
def fig3():
    ages = [("latest_full", 0, "c3_ru"), ("m2_full", 1, "c2_ja"), ("m1_full", 2, "c1_ko"), ("base_full", 3, None)]
    # 혼합 50/50은 나이가 없으므로 나이 축에 두지 않고, 각 나이 패널 오른쪽의 좁은 패널(같은 y축)에 따로 찍는다
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[4, 0.75, 1.1, 4, 0.75], wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3])]
    mix_axes = [fig.add_subplot(gs[0, 1], sharey=axes[0]), fig.add_subplot(gs[0, 4], sharey=axes[1])]
    rows = []
    for s, color, mk, ls in zip(SEEDS, (BLUE, ORANGE), ("o", "s"), ("-", "--")):
        m3 = trunk[f"trunk_fresh_s{s}"]["c3_ru"]
        e = [en_mean(branch[s][a]) for a, _, _ in ages]
        m = [mid_mean(branch[s][a], m3) for a, _, _ in ages]
        axes[0].plot(range(4), e, color=color, ls=ls, lw=2, marker=mk, ms=5, mec=SURFACE, mew=1.2, zorder=3,
                     label=f"trunk seed {s}")
        axes[1].plot(range(4), m, color=color, ls=ls, lw=2, marker=mk, ms=5, mec=SURFACE, mew=1.2, zorder=3)
        # 사후 관찰 (판정 아님): base_full + 교사(anchor)의 영어 저하
        bf = en_mean(branch[s]["base_full"])
        post = [bf + en_mean(trunk[f"trunk_fresh_s{s}"][t]) for _, _, t in ages[:3]] + [bf]
        axes[0].plot(range(4), post, color=color, ls=":", lw=1.2, marker=mk, ms=4, mfc="white", zorder=2)
        mix_e, mix_m = en_mean(branch[s]["mix"]), mid_mean(branch[s]["mix"], m3)
        dx = -0.12 if s == SEEDS[0] else 0.12
        mix_axes[0].scatter([dx], [mix_e], marker="D", s=30, color=color, edgecolor=SURFACE, zorder=4)
        mix_axes[1].scatter([dx], [mix_m], marker="D", s=30, color=color, edgecolor=SURFACE, zorder=4)
        for (a, age, _), ev, mv, pv in zip(ages, e, m, post):
            rows.append([s, a, age, round(ev, 3), round(mv, 3), round(pv, 3)])
        rows.append([s, "mix", "", round(mix_e, 3), round(mix_m, 3), ""])
    for ax in axes:
        ax.set_xticks([0, 1, 2, 3], ["0\nlatest (M_3)", "1\nM_2", "2\nM_1", "3\nbase"])
        ax.set_xlim(-0.35, 3.35)
        ax.set_xlabel("anchor age (cycles before M_3)")
        ax.xaxis.set_label_coords(0.5, -0.265)
    for ax in mix_axes:
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0], ["50/50\nmix"])
        ax.set_xlabel("(no age)")
        ax.xaxis.set_label_coords(0.5, -0.265)
        ax.tick_params(axis="y", labelleft=False)
        ax.set_facecolor("white")
    axes[0].set_ylabel("English ppl change vs base (%)")
    axes[0].set_title("(a) English: monotone in age (supported)", loc="left")
    axes[1].set_ylabel("ko/ja/ru ppl change vs M_3 (%)")
    axes[1].set_title("(b) Intermediate: not monotone", loc="left")
    axes[0].annotate("dotted, open markers:\nbase_full + anchor's own\nEnglish drift (post hoc)", (0.03, 0.05),
                     xycoords="axes fraction", fontsize=7, color=INK2)
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.9)
    save(fig, "fig3_age", rows, ["trunk_seed", "arm", "anchor_age", "english_mean_pct", "mid_mean_pct_vs_M3",
                                 "posthoc_base_plus_anchor_drift"])


# ---------------- Fig 4: P5 산점도 ----------------
def fig4():
    groups = {"trunk checkpoints (FT/fresh/base, c1-c3)": ([], BLUE, "o"),
              "C2 branches": ([], ORANGE, "s"),
              "C3/C4 branches (not in pre-specified set)": ([], AQUA, "^")}
    names = list(groups)
    for t, ev in trunk.items():
        for tag in ("c1_ko", "c2_ja", "c3_ru"):
            groups[names[0]][0].append((f"{t}/{tag}", ev[tag]))
    for s in SEEDS:
        for arm, e in branch[s].items():
            g = names[1] if arm in ("latest_full", "latest_ep", "base_full", "base_ep", "ft150") else names[2]
            groups[g][0].append((f"branch_s{s}/{arm}", e))
    p5 = check["P5"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    rows = []
    for ax, key, xl in zip(axes, ("kl_base_ckpt", "param_dist"),
                           ("probe KL(base || checkpoint), nats/token", "relative parameter distance ||θ−θ0||/||θ0||")):
        for g, (pts, color, mk) in groups.items():
            hollow = g == names[2]
            ax.scatter([e[key] for _, e in pts], [en_mean(e) for _, e in pts], s=22, marker=mk,
                       facecolor="white" if hollow else color, edgecolor=color if hollow else SURFACE,
                       linewidth=1.2, zorder=3, label=g)
        ax.set_xlabel(xl)
    for g, (pts, _, _) in groups.items():
        for n, e in pts:
            rows.append([g, n, e["kl_base_ckpt"], e["param_dist"], round(en_mean(e), 3)])
    axes[0].set_ylabel("English ppl change vs base (%)")
    axes[0].set_title(f"(a) Spearman ρ = {p5['rho_kl']:.3f}", loc="left")
    axes[1].set_title(f"(b) Spearman ρ = {p5['rho_param_dist']:.3f}", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.text(0.5, -0.04, f"ρ computed on the pre-specified set (trunks + C2, n = {p5['n']}). Checkpoints share trunks, "
                         "so points are not independent.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "fig4_p5", rows, ["group", "checkpoint", "probe_kl_base_ckpt", "param_dist", "english_mean_pct"])


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
