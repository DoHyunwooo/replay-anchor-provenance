"""
forgetting-lab / run04_repeat_check.py

LAB_NOTES.md 7장 "run04 반복 실험 판정 기준"을 그대로 계산한다 (결과 보기 전에 확정한 기준).

  * 반복 런: run04_selfgen20_seed43 (run04와 같고 seed만 43 → 데이터 순서만 다름, 복습 데이터 같음).
  * 반복 성공: 위키·책·웹 세 영역 모두 사고 없는 시드 중앙값(복습 없음) 대비 여전히 "확실히 덜 잊음"
    (6장 흔들림 기준: 위키 6.5, 책 5.0, 웹 5.0%p).
  * 반복 실패: 한 영역이라도 기준 미만 → run04 결과는 우연일 수 있음, 다음 실험 보류.
  * 참고 (판정 아님): 두 run04 시드의 최종 "원본 대비 % 변화" 차이 = 복습 있는 설정의 흔들림.
  * 참고: 반복 런의 불안정 사고(6장 정의)와 grad_norm 보조 신호도 기록한다. 판정은 최종 스텝 값으로만 한다.

결과는 콘솔과 <root>/run04_repeat_check.json 에 남긴다.

예)
  python run04_repeat_check.py
  python run04_repeat_check.py --repeat run04_selfgen20 --out <임시.json>   # 코드 점검용 (같은 런끼리 = 흔들림 0)
"""
import argparse
import json
import os
import statistics

from run04_check import (ARC, DOMAINS, MARK, NAMES, PPL_METRICS, RUN03, TARGET,
                         final_changes, load_complete, load_criteria, sign)
from seed_check import SEED_RUNS, grad_log, grad_spikes, incidents

REPEAT = "run04_selfgen20_seed43"


def main():
    p = argparse.ArgumentParser(description="run04 반복 실험 판정 (LAB_NOTES 7장 기준)")
    p.add_argument("--root", default="results/fp16")
    p.add_argument("--ckpt_root", default="checkpoints")
    p.add_argument("--first", default=TARGET, help="첫 run04 (기본 run04_selfgen20)")
    p.add_argument("--repeat", default=REPEAT, help="반복 런 (기본 run04_selfgen20_seed43)")
    p.add_argument("--out", help="결과 json (기본 <root>/run04_repeat_check.json)")
    args = p.parse_args()

    clean, thr = load_criteria(args.root)
    rdf = load_complete(args.root, args.repeat)
    first, fstep = final_changes(load_complete(args.root, args.first))
    rep, rstep = final_changes(rdf)
    r3step = final_changes(load_complete(args.root, RUN03))[1]
    if not fstep == rstep == r3step:
        raise SystemExit(f"최종 스텝이 다릅니다 ({args.first} {fstep}, {args.repeat} {rstep}, {RUN03} {r3step}). "
                         "학습·평가가 끝났는지 확인하세요.")
    seeds = {s: final_changes(load_complete(args.root, SEED_RUNS[s]))[0] for s in clean}
    med = {m: statistics.median(seeds[s][m] for s in clean) for m in first}

    # ---- 1. 판정 ----
    print(f"=== 1. 반복 판정: {args.repeat}가 세 영역 모두 복습 없음(seed {', '.join(map(str, clean))} 중앙값)보다 "
          f"확실히 덜 잊었나 (최종 스텝 {rstep})")
    dm = {m: rep[m] - med[m] for m in rep}
    sig = {m: sign(dm[m], thr[m]) for m in DOMAINS}
    for m in DOMAINS:
        print(f"  {NAMES[m]:<4} {args.repeat} {rep[m]:+.1f}% vs 기준점 {med[m]:+.1f}% → {dm[m]:+.1f}%p "
              f"(기준 {thr[m]:.1f}%p): {MARK[sig[m]]}   [참고: {args.first} {first[m] - med[m]:+.1f}%p]")
    ok = all(v == -1 for v in sig.values())
    short = [NAMES[m] for m in DOMAINS if sig[m] != -1]
    verdict = ("반복 성공: 자기 생성 복습의 보호가 다른 시드에서도 재현됨." if ok else
               f"반복 실패 ({', '.join(short)} 기준 미만): run04 결과는 우연일 수 있음. 다음 실험 보류.")
    print(f"  → {verdict}")

    # ---- 2. 참고: 복습 있는 설정의 흔들림 ----
    print(f"\n=== 2. 참고 (판정 아님): 두 run04 시드의 차이 = 복습 있는 설정의 흔들림 "
          f"({args.first} = seed 42, {args.repeat})")
    spread = {}
    for m in PPL_METRICS + [ARC]:
        spread[m] = abs(rep[m] - first[m])
        unit = "%p" if m == ARC else "%"
        print(f"  {NAMES[m]:<8} {first[m]:+.1f}{unit} vs {rep[m]:+.1f}{unit} → 차이 {spread[m]:.1f}%p"
              + (f"   (6장 복습 없음 시드 기준 {thr[m]:.1f}%p)" if m in thr else "   (7장 ARC 기준 3.0%p)"))

    # ---- 3. 참고: 사고와 보조 신호 ----
    inc = incidents(rdf)
    log, src = grad_log(args.repeat, args.ckpt_root)
    spikes = grad_spikes(log)
    print(f"\n=== 3. 참고: {args.repeat} 불안정 사고 (6장 정의): " + (", ".join(
        f"step {x['prev_step']}→{x['step']} (한국어 {x['ko_change']:+}%, 위키 {x['en_wiki_change']:+}%, "
        f"책 {x['en_books_change']:+}%, 웹 {x['en_web_change']:+}%)" for x in inc) or "없음"))
    print(f"  보조 신호 (grad_norm > 직전 50스텝 중앙값 x5, 기록 {len(log)}개, {src}): "
          + (", ".join(f"step {s['step']} ({s['ratio']}배)" for s in spikes) or "없음"))

    out = args.out or os.path.join(args.root, "run04_repeat_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"criteria": "LAB_NOTES.md 7장 run04 반복 판정 기준 (2026-09-25 확정)",
                   "first": args.first, "repeat": args.repeat, "step": rstep, "clean_seeds": clean,
                   "thresholds": thr, "changes": {args.first: first, args.repeat: rep, "clean_seed_median": med},
                   "repeat_diff_vs_median_pp": dm, "repeat_sign_vs_median": sig, "success": ok,
                   "verdict": verdict, "seed_spread_pp": spread,
                   "repeat_incidents": inc, "repeat_grad_spikes": spikes, "grad_log_source": src},
                  f, indent=2, ensure_ascii=False, default=float)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
