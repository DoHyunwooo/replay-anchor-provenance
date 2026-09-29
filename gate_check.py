"""
forgetting-lab / gate_check.py

LAB_NOTES.md 9장 "B단계 cycle 1 게이트 (수정판)"를 그대로 계산한다 (B단계 결과 보기 전 확정, 2026-09-26).
run04 반복 실험을 대체하는 게이트: cycle 1 종료 시점(c1_ko)에서 fresh 줄기 두 시드 vs FT 줄기 두 시드.

영어 위키 / 책 / 웹 각각 (값 = 원본 대비 ppl % 변화, 클수록 더 잊음):
  0. 판정 불가: FT 두 시드 평균이 5% 미만이면 "판정 불가 (기준선 망각 부족)". 실패로 치지 않는다.
  1. 방향: fresh 두 시드가 모두 FT 두 시드보다 작다 (fresh 중 큰 값 < FT 중 작은 값).
  2. 크기: FT 평균 − fresh 평균 ≥ max(2 x max(FT 범위, fresh 범위), 2.0%p).
  참고: 상대 감소율 (FT 평균 − fresh 평균) / FT 평균.

게이트:
  실패     : 판정 가능한 영역 중 하나라도 1이나 2를 못 만족 → C단계 보류, 원인 확인.
  통과     : 판정 가능한 영역이 모두 1과 2를 만족하고, 판정 가능한 영역이 2개 이상.
  판정 불가: 실패가 아니고 판정 가능한 영역이 2개 미만 → cycle 길이를 늘리는 방안 검토 (자동으로 늘리지 않음, 사용자에게 보고).
  (판정 가능한 영역이 1개뿐인데 그게 실패면 "실패"로 본다. 실패 조건이 먼저.)

결과: 콘솔 + <results_root>/gate_check.json

예)
  python gate_check.py
  python gate_check.py --results_root <가짜 결과 폴더>     # 테스트
"""
import argparse
import json
import os
import statistics

from cycle_summary import read_eval

TAG = "c1_ko"
DOMAINS = {"en_wiki": "위키", "en_books": "책", "en_web": "웹"}
FT = ["trunk_FT_s42", "trunk_FT_s43"]
FRESH = ["trunk_fresh_s42", "trunk_fresh_s43"]
MIN_BASELINE = 5.0      # FT 평균 망각(%)이 이보다 작으면 판정 불가
FLOOR = 2.0             # 크기 기준 바닥값 (%p)
MIN_JUDGEABLE = 2
EPS = 1e-9              # 부동소수점 오차로 경계값이 뒤집히지 않게


def main():
    p = argparse.ArgumentParser(description="B단계 cycle 1 게이트 (LAB_NOTES 9장 수정판)")
    p.add_argument("--results_root", default="results/cycles")
    p.add_argument("--out", help="기본: <results_root>/gate_check.json")
    args = p.parse_args()

    base = read_eval(os.path.join(args.results_root, "base", "eval.csv")).get("c0_base")
    if not base:
        raise SystemExit("원본(c0_base) 결과가 없습니다.")
    chg = {}
    for t in FT + FRESH:
        ev = read_eval(os.path.join(args.results_root, t, "eval.csv")).get(TAG)
        if not ev or any(f"ppl_{d}" not in ev for d in DOMAINS):
            raise SystemExit(f"{t}: {TAG} 영어 결과가 없습니다. B단계(trunk.py)가 끝났는지 확인하세요.")
        chg[t] = {d: (ev[f"ppl_{d}"] / base[f"ppl_{d}"] - 1) * 100 for d in DOMAINS}

    print(f"=== B단계 cycle 1 게이트 ({TAG}, 원본 대비 ppl % 변화): fresh {FRESH} vs FT {FT}")
    rows, judgeable, failed = {}, [], []
    for d, name in DOMAINS.items():
        ft, fr = [chg[t][d] for t in FT], [chg[t][d] for t in FRESH]
        ft_m, fr_m = statistics.mean(ft), statistics.mean(fr)
        diff = ft_m - fr_m
        thr = max(2 * max(max(ft) - min(ft), max(fr) - min(fr)), FLOOR)
        direction = max(fr) < min(ft)
        size = diff >= thr - EPS           # 경계값(차이 = 기준)은 통과
        rel = diff / ft_m if ft_m > 0 else None
        if ft_m < MIN_BASELINE - EPS:      # 정확히 5.0%면 판정 가능
            status = "판정 불가 (기준선 망각 부족)"
        else:
            judgeable.append(name)
            status = "통과" if direction and size else "실패"
            if status == "실패":
                failed.append(name)
        rows[d] = {"ft": ft, "fresh": fr, "ft_mean": ft_m, "fresh_mean": fr_m, "diff_pp": diff,
                   "threshold_pp": thr, "direction_ok": direction, "size_ok": size,
                   "relative_reduction": rel, "status": status}
        print(f"  {name}: FT {ft[0]:+.1f} / {ft[1]:+.1f}% (평균 {ft_m:+.1f}%), fresh {fr[0]:+.1f} / {fr[1]:+.1f}% "
              f"(평균 {fr_m:+.1f}%)")
        print(f"      방향 {'O' if direction else 'X'} (fresh 최대 {max(fr):+.1f} {'<' if direction else '≥'} "
              f"FT 최소 {min(ft):+.1f}), "
              f"크기 {'O' if size else 'X'} (차이 {diff:+.1f}%p vs 기준 {thr:.1f}%p), "
              f"상대 감소율 {'-' if rel is None else f'{rel:.0%}'} (참고) → {status}")

    if failed:
        verdict = f"실패 ({', '.join(failed)}): C단계 보류, 원인 확인."
    elif len(judgeable) >= MIN_JUDGEABLE:
        verdict = f"통과 (판정 가능한 영역 {', '.join(judgeable)} 모두 방향·크기 만족)."
    else:
        verdict = (f"판정 불가 (판정 가능한 영역 {len(judgeable)}개 < {MIN_JUDGEABLE}): "
                   "C단계 전에 cycle 길이를 늘리는 방안 검토. 자동으로 늘리지 않고 사용자에게 보고.")
    print(f"  → 게이트: {verdict}")

    out = args.out or os.path.join(args.results_root, "gate_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"criteria": "LAB_NOTES.md 9장 B단계 cycle 1 게이트 수정판 (2026-09-26 확정)", "tag": TAG,
                   "ft": FT, "fresh": FRESH, "changes_pct": chg, "domains": rows,
                   "judgeable": judgeable, "failed": failed, "verdict": verdict},
                  f, indent=2, ensure_ascii=False)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
