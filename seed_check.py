"""
forgetting-lab / seed_check.py

LAB_NOTES.md 6장 "run01 시드 반복" 판정 기준을 그대로 계산한다 (결과 보기 전에 확정한 기준).
사람이 표를 보고 세다가 틀리지 않게, 판정에 쓰는 숫자와 결론을 모두 여기서 낸다.

  1. 불안정 사고 (주 기준): 직전 체크포인트 대비 한국어 ppl +5% 이상 AND 영어 위키·책·웹 모두 +20% 이상.
     각 런의 저장 간격 그대로 판정하고, 비교용으로 50스텝 간격 부분표본 결과도 같이 보여준다.
  2. 보조 신호: grad_norm이 직전 50스텝 이동 중앙값의 5배를 넘은 스텝 (판정에는 안 씀,
     체크포인트 사이 사고를 놓쳤는지 점검용).
  3. 흔들림 기준: 사고 없는 시드들의 최종 "원본 대비 % 변화" 범위 x 2, 최소 5%p.
     사고 없는 시드가 1개 이하면 5%p.
  4. 사고 빈도: 시드 3개 중 사고 난 시드 수 (1개 이하 = 드문 사건, 2개 이상 = 흔한 사건). 거친 1차 확인.
  5. run03 재판정: 기준점 = 사고 없는 시드들의 중앙값. 3장 가설 A/B/기각 규칙을 새 기준으로 적용.
     사고 없는 시드가 0개면 재판정하지 않는다.
  6. 발견 2 확인: 세 시드 모두 최종 LAMBADA ppl 변화가 +25~35% 안인가.

결과는 콘솔과 <root>/seed_check.json 에 남긴다.

예)
  python seed_check.py
"""
import argparse
import json
import os
import statistics

import pandas as pd

from summarize import load_run

SEED_RUNS = {42: "run01_ko_wiki", 43: "run01_seed43", 44: "run01_seed44"}
TARGET = "run03_replay20"
DOMAINS = ["en_wiki_ppl", "en_books_ppl", "en_web_ppl"]
NOISE_METRICS = ["lambada_ppl"] + DOMAINS + ["ko_ppl"]

KO_JUMP, EN_JUMP = 0.05, 0.20          # 사고 정의
GRAD_FACTOR, GRAD_WINDOW, GRAD_MIN_N = 5.0, 50, 5   # 보조 신호: 직전 50스텝, 값 5개 이상일 때만
MIN_THRESHOLD = 5.0                    # 흔들림 기준 하한 (%p)
FINDING2_RANGE = (25.0, 35.0)


# ---------- 1. 불안정 사고 ----------

def incidents(df):
    """직전 체크포인트 대비 사고 조건을 만족한 스텝 목록과 각 변화율."""
    d = df.sort_values("step").reset_index(drop=True)
    out = []
    for i in range(1, len(d)):
        prev, cur = d.iloc[i - 1], d.iloc[i]
        ko = cur["ko_ppl"] / prev["ko_ppl"] - 1
        en = {c: cur[c] / prev[c] - 1 for c in DOMAINS}
        if ko >= KO_JUMP and all(v >= EN_JUMP for v in en.values()):
            out.append({"step": int(cur["step"]), "prev_step": int(prev["step"]),
                        "ko_change": round(ko * 100, 1),
                        **{c.replace("_ppl", "_change"): round(v * 100, 1) for c, v in en.items()}})
    return out


# ---------- 2. 보조 신호: grad_norm ----------

def grad_log(run, ckpt_root):
    """train_log.jsonl (매 스텝 기록)이 있으면 그걸, 없으면 마지막 체크포인트의 trainer_state.json."""
    path = os.path.join(ckpt_root, run, "train_log.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        src = "train_log.jsonl"
    elif not os.path.isdir(os.path.join(ckpt_root, run)):
        return [], "없음"
    else:
        states = [os.path.join(ckpt_root, run, d, "trainer_state.json")
                  for d in os.listdir(os.path.join(ckpt_root, run)) if d.startswith("checkpoint-")]
        states = [s for s in states if os.path.exists(s)]
        if not states:
            return [], "없음"
        last = max(states, key=lambda s: int(s.split("checkpoint-")[1].split(os.sep)[0]))
        with open(last, encoding="utf-8") as f:
            rows = json.load(f)["log_history"]
        src = os.path.relpath(last)
    return [(int(r["step"]), float(r["grad_norm"])) for r in rows if "grad_norm" in r], src


def grad_spikes(log):
    spikes = []
    for i, (step, g) in enumerate(log):
        window = [v for s, v in log[:i] if step - GRAD_WINDOW <= s < step]
        if len(window) >= GRAD_MIN_N:
            med = statistics.median(window)
            if g > GRAD_FACTOR * med:
                spikes.append({"step": step, "grad_norm": round(g, 3),
                               "moving_median": round(med, 3), "ratio": round(g / med, 1)})
    return spikes


# ---------- 공통 ----------

def final_change(df):
    """최종 스텝의 원본 대비 % 변화."""
    base, last = df[df["step"] == 0].iloc[0], df.sort_values("step").iloc[-1]
    return {m: (last[m] / base[m] - 1) * 100 for m in NOISE_METRICS}, int(last["step"])


def main():
    p = argparse.ArgumentParser(description="run01 시드 반복 판정 (LAB_NOTES 6장 기준)")
    p.add_argument("--root", default="results/fp16")
    p.add_argument("--ckpt_root", default="checkpoints")
    args = p.parse_args()

    runs = {}
    for seed, run in SEED_RUNS.items():
        df = load_run(args.root, run)
        if df.empty or df[DOMAINS + ["ko_ppl"]].isna().any().any():
            raise SystemExit(f"{run}: 결과가 없거나 비어 있는 지표가 있습니다. eval_all.py부터 돌리세요.")
        runs[seed] = df
    report = {"criteria": "LAB_NOTES.md 6장 (2026-09-25 확정)", "seeds": {}}

    # 1·2. 사고와 보조 신호
    print("=== 1. 불안정 사고 (주 기준: 한국어 +5% 이상 AND 영어 세 영역 모두 +20% 이상, 직전 체크포인트 대비)")
    clean, hit = [], []
    for seed, df in runs.items():
        steps = sorted(df["step"])
        interval = steps[2] - steps[1] if len(steps) > 2 else None
        inc = incidents(df)
        inc50 = incidents(df[df["step"] % 50 == 0])
        log, src = grad_log(SEED_RUNS[seed], args.ckpt_root)
        spikes = grad_spikes(log)
        (hit if inc else clean).append(seed)
        report["seeds"][seed] = {"run": SEED_RUNS[seed], "save_interval": interval,
                                 "incidents": inc, "incidents_50step_subsample": inc50,
                                 "grad_log_source": src, "grad_log_points": len(log),
                                 "grad_spikes": spikes}
        print(f"  seed {seed} ({SEED_RUNS[seed]}, 저장 간격 {interval}): "
              + (", ".join(f"step {x['prev_step']}→{x['step']} (한국어 {x['ko_change']:+}%, "
                           f"위키 {x['en_wiki_change']:+}%, 책 {x['en_books_change']:+}%, "
                           f"웹 {x['en_web_change']:+}%)" for x in inc) or "사고 없음"))
        if interval != 50:
            print(f"      50스텝 부분표본으로 보면: "
                  + (", ".join(f"step {x['step']}" for x in inc50) or "사고 없음") + " (참고)")
        print(f"      보조 신호 (grad_norm > 직전 50스텝 중앙값 x5, 기록 {len(log)}개, {src}): "
              + (", ".join(f"step {s['step']} ({s['ratio']}배)" for s in spikes) or "없음"))

    # 4. 빈도
    n_hit = len(hit)
    freq = "드문 사건" if n_hit <= 1 else "흔한 사건 (warmup 20 설정에서)"
    print(f"\n=== 2. 사고 빈도: 시드 3개 중 {n_hit}개 ({', '.join(map(str, hit)) or '-'}) → {freq}"
          " (시드 3개라 거친 1차 확인)")
    report["frequency"] = {"incident_seeds": hit, "clean_seeds": clean, "verdict": freq}

    # 3. 흔들림 기준
    changes = {seed: final_change(df)[0] for seed, df in runs.items()}
    print(f"\n=== 3. 흔들림 기준 (사고 없는 시드 {clean or '없음'}로 계산, 2 x 범위, 최소 {MIN_THRESHOLD}%p)")
    thresholds = {}
    for m in NOISE_METRICS:
        vals = [changes[s][m] for s in clean]
        rng = max(vals) - min(vals) if len(vals) >= 2 else None
        thr = max(MIN_THRESHOLD, 2 * rng) if rng is not None else MIN_THRESHOLD
        all_rng = max(changes[s][m] for s in runs) - min(changes[s][m] for s in runs)
        thresholds[m] = thr
        print(f"  {m:<13} 시드별 " + " / ".join(f"{s}: {changes[s][m]:+.1f}%" for s in runs)
              + f" | 사고 없는 시드 범위 {'-' if rng is None else f'{rng:.1f}%p'}"
              + f" → 기준 {thr:.1f}%p (참고: 전체 범위 {all_rng:.1f}%p)")
    fallback = len(clean) <= 1
    report["noise"] = {"final_changes": changes, "thresholds": thresholds,
                       "fallback_5pp": fallback}

    # 5. run03 재판정
    print(f"\n=== 4. {TARGET} 재판정 (기준점 = 사고 없는 시드들의 중앙값)")
    if not clean:
        print("  사고 없는 시드가 없어서 재판정하지 않음 (기준에 따라 이 사실만 기록)")
        report["rejudge"] = {"done": False, "reason": "사고 없는 시드 없음"}
    else:
        tdf = load_run(args.root, TARGET)
        tchg, tstep = final_change(tdf)
        ref = {m: statistics.median(changes[s][m] for s in clean) for m in NOISE_METRICS}
        diff = {m: tchg[m] - ref[m] for m in NOISE_METRICS}
        sig = {m: (-1 if diff[m] <= -thresholds[m] else 1 if diff[m] >= thresholds[m] else 0)
               for m in NOISE_METRICS}
        for m in NOISE_METRICS:
            mark = {-1: "확실히 덜 잊음", 1: "확실히 더 잊음", 0: "차이 없음"}[sig[m]]
            print(f"  {m:<13} {TARGET}@{tstep} {tchg[m]:+.1f}% vs 기준점 {ref[m]:+.1f}% "
                  f"→ {diff[m]:+.1f}%p (기준 {thresholds[m]:.1f}%p): {mark}")
        a = sig["en_wiki_ppl"] == -1 and sig["en_books_ppl"] >= 0
        b = sig["en_books_ppl"] == 1
        rej = all(sig[c] == -1 for c in DOMAINS)
        if a and b:
            verdict = "원래 가설 전체 지지. 복습은 복습한 영역만 지키고, 다른 문체는 오히려 밀어낸다."
        elif a:
            verdict = "가설 A만 지지: 복습은 복습한 영역만 지킨다. LAMBADA 악화 원인은 미해결."
        elif b:
            verdict = "가설 B만 지지 (밀어냄)."
        elif rej:
            verdict = "기각: 세 영역 모두 확실히 덜 잊음. LAMBADA 악화는 다른 이유로 본다."
        else:
            verdict = "가설 불지지, 결론 보류."
        if fallback:
            verdict += " [약한 근거: 사고 없는 시드가 1개뿐이라 흔들림 기준이 5%p 기본값]"
        print(f"  가설 A {'지지' if a else '불지지'}, 가설 B {'지지' if b else '불지지'}, "
              f"기각 조건 {'해당' if rej else '해당 안 됨'}\n  → {verdict}")
        report["rejudge"] = {"done": True, "target_step": tstep, "target_changes": tchg,
                             "reference_median": ref, "diff_pp": diff, "a": a, "b": b,
                             "reject": rej, "verdict": verdict}

    # 6. 발견 2
    lo, hi = FINDING2_RANGE
    lam = {s: changes[s]["lambada_ppl"] for s in runs}
    ok = all(lo <= v <= hi for v in lam.values())
    print(f"\n=== 5. 발견 2 확인: 최종 LAMBADA ppl 변화 "
          + " / ".join(f"{s}: {v:+.1f}%" for s, v in lam.items())
          + f" → {'모두 +25~35% 안: 유지' if ok else '범위 밖 있음: 발견 2 수정 필요'}")
    report["finding2"] = {"lambada_changes": lam, "all_in_range": ok}

    out = os.path.join(args.root, "seed_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
