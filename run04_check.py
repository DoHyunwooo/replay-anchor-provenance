"""
forgetting-lab / run04_check.py

LAB_NOTES.md 7장 "run04 사전 예측 v2"를 그대로 계산한다 (결과 보기 전에 확정한 기준).
seed_check.py처럼 판정에 쓰는 숫자와 결론을 모두 여기서 낸다.

  * 비교 시점: 각 런의 최종 스텝 (run04 375, run03 375, 시드 300).
  * 비교 기준: run03, 그리고 사고 없는 시드들의 중앙값 (6장). 둘 다 "원본 대비 % 변화"의 차이(%p).
  * 흔들림 기준: 6장 값 (seed_check.json에서 읽고, 7장에 적어 둔 값과 같은지 확인한다).
    위키 6.5, 책 5.0, 웹 5.0, 한국어 11.6, LAMBADA 25.5 (%p).
  1. 예측 1: 책과 웹 모두 |run04 − run03| < 기준이면 지지.
  2. 예측 2: 위키 run04 − run03 ≥ +6.5%p (run03보다 확실히 더 잊음)이면 지지.
  3. 사고 없는 시드 중앙값 대비: 지표마다 확실히 덜 잊음 / 차이 없음 / 확실히 더 잊음.
  4. 보조: 한국어 ppl (기준 11.6%p), ARC-Easy acc (기준 3%p, 7장. acc는 높을수록 덜 잊음).
  5. 참고: LAMBADA ppl (기준 25.5%p, 발견 7).
  6. 위험 점검: 6장과 같은 정의의 불안정 사고, grad_norm 보조 신호, 그리고 영역 중 하나라도
     사고 없는 시드 중앙값보다 확실히 더 잊었는지 (= 복습 없는 것보다 나쁨).
  7. 깨끗한 부분집합 평가가 필요한지: 품질 보고서의 held-out 겹침 블록이 하나라도 있으면 필요 (7장 규칙).

결과는 콘솔과 <root>/run04_check.json 에 남긴다.

예)
  python run04_check.py
  python run04_check.py --target run03_replay20 --out <임시.json>   # 코드 점검용 (run03끼리 비교 = 차이 0)
"""
import argparse
import json
import os
import statistics

from seed_check import SEED_RUNS, grad_log, grad_spikes, incidents
from summarize import load_run

TARGET = "run04_selfgen20"
RUN03 = "run03_replay20"
REPLAY_REPORT = "data/selfgen_replay/quality_report.json"

DOMAINS = ["en_wiki_ppl", "en_books_ppl", "en_web_ppl"]
PPL_METRICS = DOMAINS + ["ko_ppl", "lambada_ppl"]
ARC = "arc_easy_acc"
# 7장에 적어 둔 흔들림 기준 (%p). seed_check.json 값과 소수 첫째 자리까지 같아야 한다.
PREREG = {"en_wiki_ppl": 6.5, "en_books_ppl": 5.0, "en_web_ppl": 5.0, "ko_ppl": 11.6, "lambada_ppl": 25.5}
# ARC 기준 (%p, 7장 2026-09-25 확정, run04 결과 보기 전). 측정 오차 ±1.03%p → 두 결과 비교 2σ ≈ 2.9%p.
# 사고 없는 시드 범위 기반(2 x 1.2 = 2.4%p)보다 측정 오차가 더 믿을 만한 바닥이라 이걸 쓴다.
ARC_THRESHOLD = 3.0
NAMES = {"en_wiki_ppl": "위키", "en_books_ppl": "책", "en_web_ppl": "웹", "ko_ppl": "한국어",
         "lambada_ppl": "LAMBADA", ARC: "ARC-Easy"}


def final_changes(df):
    """최종 스텝의 원본 대비 변화: ppl은 % 변화, ARC acc는 %p 차이."""
    base, last = df[df["step"] == 0].iloc[0], df.sort_values("step").iloc[-1]
    out = {m: (last[m] / base[m] - 1) * 100 for m in PPL_METRICS}
    out[ARC] = (last[ARC] - base[ARC]) * 100
    return out, int(last["step"])


def sign(diff, thr):
    """-1 = 확실히 덜 잊음, 1 = 확실히 더 잊음, 0 = 차이 없음 (ppl 기준: 높을수록 더 잊음)."""
    return -1 if diff <= -thr else 1 if diff >= thr else 0


MARK = {-1: "확실히 덜 잊음", 0: "차이 없음", 1: "확실히 더 잊음"}


def load_criteria(root):
    """6장 결과(seed_check.json)에서 사고 없는 시드와 흔들림 기준. 7장에 적은 값과 다르면 멈춘다."""
    with open(os.path.join(root, "seed_check.json"), encoding="utf-8") as f:
        sc = json.load(f)
    clean = [int(s) for s in sc["frequency"]["clean_seeds"]]
    thr = {m: sc["noise"]["thresholds"][m] for m in PREREG}
    bad = {m: (round(thr[m], 1), v) for m, v in PREREG.items() if round(thr[m], 1) != v}
    if bad:
        raise SystemExit(f"seed_check.json의 흔들림 기준이 7장에 적은 값과 다릅니다: {bad}")
    if not clean:
        raise SystemExit("사고 없는 시드가 없습니다 (6장). 중앙값 기준점을 만들 수 없습니다.")
    return clean, thr


def load_complete(root, run):
    df = load_run(root, run)
    if df.empty or df[DOMAINS + ["ko_ppl", "lambada_ppl", ARC]].isna().any().any():
        raise SystemExit(f"{run}: 결과가 없거나 비어 있는 지표가 있습니다. eval_all.py부터 돌리세요.")
    return df


def main():
    p = argparse.ArgumentParser(description="run04 사전 예측 판정 (LAB_NOTES 7장 기준)")
    p.add_argument("--root", default="results/fp16")
    p.add_argument("--ckpt_root", default="checkpoints")
    p.add_argument("--target", default=TARGET, help="판정할 런 (기본 run04_selfgen20)")
    p.add_argument("--out", help="결과 json (기본 <root>/run04_check.json, 다른 런이면 <root>/<런>_check.json)")
    args = p.parse_args()

    clean, thr = load_criteria(args.root)
    load = lambda run: load_complete(args.root, run)

    tdf, r3df = load(args.target), load(RUN03)
    t, tstep = final_changes(tdf)
    r3, r3step = final_changes(r3df)
    if tstep != r3step:
        raise SystemExit(f"{args.target} 최종 스텝 {tstep} ≠ {RUN03} {r3step}. 학습·평가가 끝났는지 확인하세요.")
    seeds = {s: final_changes(load(SEED_RUNS[s]))[0] for s in clean}
    med = {m: statistics.median(seeds[s][m] for s in clean) for m in t}
    report = {"criteria": "LAB_NOTES.md 7장 사전 예측 v2 (2026-09-25 확정)", "target": args.target,
              "step": tstep, "thresholds": thr, "clean_seeds": clean,
              "changes": {args.target: t, RUN03: r3, "clean_seed_median": med}}

    d3 = {m: t[m] - r3[m] for m in t}      # run04 − run03
    dm = {m: t[m] - med[m] for m in t}     # run04 − 시드 중앙값

    # ---- 1. 예측 1 ----
    print(f"=== 1. 예측 1: 책·웹은 run03과 비슷하게 지킨다 (|{args.target} − run03| < 기준), 최종 스텝 {tstep}")
    p1 = {}
    for m in ["en_books_ppl", "en_web_ppl"]:
        s = sign(d3[m], thr[m])
        p1[m] = s == 0
        print(f"  {NAMES[m]:<4} {args.target} {t[m]:+.1f}% vs run03 {r3[m]:+.1f}% → {d3[m]:+.1f}%p "
              f"(기준 {thr[m]:.1f}%p): {'비슷함' if s == 0 else 'run03보다 ' + MARK[s]}")
    p1_ok = all(p1.values())
    print(f"  → 예측 1 {'지지' if p1_ok else '불지지'}")

    # ---- 2. 예측 2 ----
    m = "en_wiki_ppl"
    s_wiki = sign(d3[m], thr[m])
    p2_ok = s_wiki == 1
    print(f"\n=== 2. 예측 2: 위키는 run03보다 확실히 덜 지킨다 ({args.target} − run03 ≥ +{thr[m]:.1f}%p)")
    print(f"  위키 {args.target} {t[m]:+.1f}% vs run03 {r3[m]:+.1f}% → {d3[m]:+.1f}%p: "
          f"{'run03과 ' + MARK[s_wiki] if s_wiki == 0 else 'run03보다 ' + MARK[s_wiki]}")
    print(f"  → 예측 2 {'지지' if p2_ok else '불지지'}"
          + (" (반대 방향: run03보다 위키를 더 잘 지킴)" if s_wiki == -1 else ""))

    # ---- 3. 시드 중앙값 대비 ----
    print(f"\n=== 3. 사고 없는 시드 중앙값 대비 (seed {', '.join(map(str, clean))}, 복습 없음)")
    vs_med = {}
    for m in PPL_METRICS:
        vs_med[m] = sign(dm[m], thr[m])
        print(f"  {NAMES[m]:<7} {args.target} {t[m]:+.1f}% vs 기준점 {med[m]:+.1f}% → {dm[m]:+.1f}%p "
              f"(기준 {thr[m]:.1f}%p): {MARK[vs_med[m]]}" + ("  [참고]" if m == "lambada_ppl" else ""))
    protected = [NAMES[m] for m in DOMAINS if vs_med[m] == -1]
    print(f"  → 복습 없는 것보다 확실히 덜 잊은 영역: {', '.join(protected) or '없음'}")

    # ---- 4. 보조 ----
    m = "ko_ppl"
    s_ko = sign(d3[m], thr[m])
    print(f"\n=== 4. 보조")
    print(f"  한국어 ppl {args.target} {t[m]:+.1f}% vs run03 {r3[m]:+.1f}% → {d3[m]:+.1f}%p "
          f"(기준 {thr[m]:.1f}%p): {'차이 없음' if s_ko == 0 else ('run03보다 한국어를 덜 배움' if s_ko == 1 else 'run03보다 한국어를 더 배움')}")
    # acc는 높을수록 덜 잊음 → 부호를 뒤집어 ppl과 같은 뜻(-1 = 덜 잊음)으로 맞춘다
    arc_r3, arc_med = sign(-d3[ARC], ARC_THRESHOLD), sign(-dm[ARC], ARC_THRESHOLD)
    print(f"  ARC-Easy acc {args.target} {t[ARC]:+.1f}%p vs run03 {r3[ARC]:+.1f}%p → {d3[ARC]:+.1f}%p "
          f"(기준 {ARC_THRESHOLD:.1f}%p): {'차이 없음' if arc_r3 == 0 else 'run03보다 ' + MARK[arc_r3]}")
    print(f"  ARC-Easy acc {args.target} {t[ARC]:+.1f}%p vs 시드 중앙값 {med[ARC]:+.1f}%p → {dm[ARC]:+.1f}%p "
          f"(기준 {ARC_THRESHOLD:.1f}%p): {MARK[arc_med]}")

    # ---- 5. 참고 ----
    m = "lambada_ppl"
    print(f"\n=== 5. 참고: LAMBADA ppl {args.target} {t[m]:+.1f}% vs run03 {r3[m]:+.1f}% → {d3[m]:+.1f}%p "
          f"(기준 {thr[m]:.1f}%p): {'차이 없음' if sign(d3[m], thr[m]) == 0 else MARK[sign(d3[m], thr[m])]}")

    # ---- 6. 위험 점검 ----
    print(f"\n=== 6. 위험 점검")
    inc = incidents(tdf)
    log, src = grad_log(args.target, args.ckpt_root)
    spikes = grad_spikes(log)
    worse = [NAMES[m] for m in DOMAINS if vs_med[m] == 1]
    print("  불안정 사고 (6장 정의): " + (", ".join(
        f"step {x['prev_step']}→{x['step']} (한국어 {x['ko_change']:+}%, 위키 {x['en_wiki_change']:+}%, "
        f"책 {x['en_books_change']:+}%, 웹 {x['en_web_change']:+}%)" for x in inc) or "없음"))
    print(f"  보조 신호 (grad_norm > 직전 50스텝 중앙값 x5, 기록 {len(log)}개, {src}): "
          + (", ".join(f"step {s['step']} ({s['ratio']}배)" for s in spikes) or "없음"))
    print(f"  복습 없는 것보다 확실히 더 잊은 영역: {', '.join(worse) or '없음'}"
          + (" → 위험 현실화 (복습이 오히려 악화)" if worse else ""))
    run04b = [NAMES[m] for m in DOMAINS if sign(d3[m], thr[m]) == 1]
    print(f"  run03보다 확실히 더 잊은 영역: {', '.join(run04b) or '없음'}"
          + (" → run04b(코드·망가진 블록 거른 버전) 후보 조건 해당 (7장)" if run04b else ""))

    # ---- 7. 깨끗한 부분집합 ----
    need_clean = None
    if os.path.exists(REPLAY_REPORT):
        with open(REPLAY_REPORT, encoding="utf-8") as f:
            ov = json.load(f)[0]["overlap"]
        n = {d: ov[d]["blocks_with_overlap"] for d in ov}
        need_clean = any(n.values())
        print(f"\n=== 7. held-out 겹침 (품질 보고서): " + ", ".join(f"{d} {v}/400" for d, v in n.items())
              + (" → 깨끗한 부분집합 평가 필요 (clean_subset_eval.py --replay data/selfgen_replay)"
                 if need_clean else " → 깨끗한 부분집합 평가 불필요"))
    else:
        print(f"\n=== 7. {REPLAY_REPORT} 없음: 겹침 확인 불가")

    report.update({
        "diff_vs_run03_pp": d3, "diff_vs_median_pp": dm,
        "prediction1": {"books_similar": p1["en_books_ppl"], "web_similar": p1["en_web_ppl"], "supported": p1_ok},
        "prediction2": {"wiki_sign_vs_run03": s_wiki, "supported": p2_ok},
        "vs_median_sign": vs_med, "ko_sign_vs_run03": s_ko,
        "arc": {"threshold_pp": ARC_THRESHOLD, "sign_vs_run03": arc_r3, "sign_vs_median": arc_med},
        "risk": {"incidents": inc, "grad_log_source": src, "grad_spikes": spikes,
                 "worse_than_no_replay": worse, "worse_than_run03": run04b},
        "clean_subset_needed": need_clean,
    })
    # 다른 런을 넣었을 때 run04 결과 파일을 덮어쓰지 않게
    out = args.out or os.path.join(args.root, "run04_check.json" if args.target == TARGET
                                   else f"{args.target}_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
