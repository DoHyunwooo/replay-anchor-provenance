"""
forgetting-lab / c_check.py

LAB_NOTES.md 9장 "C단계 판정 기준"을 그대로 계산한다 (C단계 결과 보기 전 확정, 2026-09-26).
있는 결과만 판정하고, 입력이 없는 예측은 "결과 없음"으로 적는다 (C1만 끝났을 때도 돌릴 수 있게).

지표
  원래 지식 : 영어 wiki/books/web ppl의 원본 대비 % 변화 (영역별 판정). 클수록 더 잊음.
  중간 지식 : 분기 = ko/ja/ru held-out ppl의 분기 종료 값의 M_3 대비 % 변화.
              C1(줄기) = 그 언어를 배운 cycle 직후 대비 c3의 % 변화 (ko: c1→c3, ja: c2→c3). 클수록 더 잊음.
  새 지식   : 그리스어 held-out ppl의 M_3 대비 % 변화 (음수 = 감소 = 배움). 보고만 (예측 없음).
흔들림 기준 (수정판 2026-09-27, 지표와 같은 종류의 양으로 계산. B단계 fresh 줄기 두 시드의 범위 x 2, 최소 2.0%p):
  영어 (영역별)  : c3의 원본 대비 % 변화.
  새 지식 (그리스어): 각 cycle에 배운 언어가 그 cycle 동안 변한 % (ko c0→c1, ja c1→c2, ru c2→c3) 셋의 기준 중 최댓값.
  중간 지식      : 이미 배운 언어의 다음 한 cycle 동안 % 변화 (ko c2→c3, ja c2→c3). 러시아어는 대응값이 없어 ko·ja 중 큰 값.
  C3 양 끝 비교용 평균 지표 (같은 원칙): 영어 세 영역 평균의 c3 원본 대비 %, 중간 = (ko c2→c3, ja c2→c3) 평균.
  B단계는 기준 보정용, C단계는 검정용 (기준을 C단계 결과로 다시 계산하지 않음).
  (이전 판: 모든 지표를 "c3 원본 대비 % 변화" 범위로 계산 → 정의 불일치로 대체. 그리스어 16.3%p 폐기.)
"확실한 차이": 두 줄기(C1은 두 시드)의 쌍별 차이 d = A − B가 모두 같은 부호(0 아님)이고 |평균 d| ≥ 기준.
  줄기별 값, 쌍별 차이, 평균, 효과 크기(평균 차이 / 기준, 상대 차이 = 평균 차이 / |B 평균|)를 모두 남긴다.
여러 지표를 묶은 예측의 판정 (9장, 결과 전 확정):
  지지 = 모든 지표가 예측 방향으로 확실 / 부분 지지 = 예측 방향 확실 1개 이상, 반대 방향 확실 없음 /
  반대 = 반대 방향 확실 1개 이상, 예측 방향 확실 없음 / 혼재 = 양쪽 다 있음 / 불지지 = 확실한 차이 없음.

예측: C1-P1, C1-P2, P1, P2, P3, P4, P5, FT-150 비교(탐색), C3 진행 조건, C3 예측(단조).
결과: 콘솔 + <results_root>/c_check.json

예)
  python c_check.py
  python c_check.py --results_root <가짜 결과 폴더>      # 테스트
"""
import argparse
import json
import os
import statistics

from cycle_summary import read_eval

EN = ["en_wiki", "en_books", "en_web"]
MID = ["ko", "ja", "ru"]
SEEDS = [42, 43]
FLOOR = 2.0
EPS = 1e-9
FRESH = {s: f"trunk_fresh_s{s}" for s in SEEDS}
BASET = {s: f"trunk_base_s{s}" for s in SEEDS}
LEARNED_AT = {"ko": "c1_ko", "ja": "c2_ja"}          # C1 중간 지식: 배운 cycle
AGE = {"latest_full": 0, "m2_full": 1, "m1_full": 2, "base_full": 3}
NAMES = {"en_wiki": "위키", "en_books": "책", "en_web": "웹", "ko": "한국어", "ja": "일본어", "ru": "러시아어",
         "el": "그리스어"}


def pct(a, b):
    return (a / b - 1) * 100


def compare(A, B, thr, predicted=None):
    """A, B: 쌍 이름 -> 값. predicted: -1 (A가 작다 예측) / +1 / None. 반환 dict."""
    keys = [k for k in A if k in B]
    d = {k: A[k] - B[k] for k in keys}
    mean_d = statistics.mean(d.values())
    same = all(v > 0 for v in d.values()) or all(v < 0 for v in d.values())
    certain = same and abs(mean_d) >= thr - EPS
    sign = (1 if mean_d > 0 else -1) if certain else 0
    b_mean = statistics.mean(B[k] for k in keys)
    r = {"A": A, "B": B, "diffs": d, "mean_diff": mean_d, "threshold": thr, "same_direction": same,
         "certain": certain, "sign": sign, "effect_over_threshold": mean_d / thr,
         "relative_diff": mean_d / abs(b_mean) if b_mean else None}
    if predicted is not None:
        r["outcome"] = ("예측 방향 확실" if sign == predicted else "반대 방향 확실" if sign else "확실한 차이 없음")
    return r


def aggregate(results):
    pro = sum(r["outcome"] == "예측 방향 확실" for r in results.values())
    con = sum(r["outcome"] == "반대 방향 확실" for r in results.values())
    if pro == len(results):
        return "지지"
    if pro and con:
        return "혼재"
    if pro:
        return "부분 지지"
    if con:
        return "반대"
    return "불지지"


def show(title, results, fmt="{:+.1f}"):
    print(f"  {title}")
    for m, r in results.items():
        per = ", ".join(f"{k}: {fmt.format(r['A'][k])} vs {fmt.format(r['B'][k])} (d {r['diffs'][k]:+.1f})"
                        for k in r["diffs"])
        print(f"    {NAMES.get(m, m)}: {per} | 평균 차이 {r['mean_diff']:+.2f}%p, 기준 {r['threshold']:.1f}, "
              f"같은 방향 {'O' if r['same_direction'] else 'X'}, 효과 {r['effect_over_threshold']:+.2f}x기준"
              + (f" → {r['outcome']}" if "outcome" in r else f" → {'확실' if r['certain'] else '확실한 차이 없음'}"))


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def main():
    p = argparse.ArgumentParser(description="C단계 판정 (LAB_NOTES 9장)")
    p.add_argument("--results_root", default="results/cycles")
    p.add_argument("--out", help="기본: <results_root>/c_check.json")
    args = p.parse_args()
    R = args.results_root
    base = read_eval(os.path.join(R, "base", "eval.csv")).get("c0_base")
    if not base:
        raise SystemExit("원본(c0_base) 결과가 없습니다.")
    trunk = {t: read_eval(os.path.join(R, t, "eval.csv")) for t in list(FRESH.values()) + list(BASET.values())
             + [f"trunk_FT_s{s}" for s in SEEDS]}
    branch = {s: read_eval(os.path.join(R, "branches", FRESH[s], "eval.csv")) for s in SEEDS}
    report = {"criteria": "LAB_NOTES.md 9장 C단계 판정 기준 (2026-09-26 확정)"}

    # ---- 흔들림 기준 (수정판: 지표와 같은 종류의 양, fresh 줄기 두 시드 범위 x 2, 최소 2.0%p) ----
    tags = ("c1_ko", "c2_ja", "c3_ru")
    if not all(trunk[FRESH[s]].get(t) for s in SEEDS for t in tags):
        raise SystemExit("fresh 줄기 c1~c3 결과가 없습니다 (흔들림 기준 계산 불가).")
    f3 = {s: trunk[FRESH[s]]["c3_ru"] for s in SEEDS}
    ppl = lambda s, tag, m: base[f"ppl_{m}"] if tag == "c0" else trunk[FRESH[s]][tag][f"ppl_{m}"]
    spread = lambda fn: 2 * abs(fn(SEEDS[0]) - fn(SEEDS[1]))
    raw = {}
    for d in EN:
        raw[d] = spread(lambda s: pct(ppl(s, "c3_ru", d), base[f"ppl_{d}"]))
    learn = {"ko": ("c0", "c1_ko"), "ja": ("c1_ko", "c2_ja"), "ru": ("c2_ja", "c3_ru")}
    raw["el_parts"] = {l: spread(lambda s: pct(ppl(s, b, l), ppl(s, a, l))) for l, (a, b) in learn.items()}
    raw["el"] = max(raw["el_parts"].values())
    raw["ko"] = spread(lambda s: pct(ppl(s, "c3_ru", "ko"), ppl(s, "c2_ja", "ko")))
    raw["ja"] = spread(lambda s: pct(ppl(s, "c3_ru", "ja"), ppl(s, "c2_ja", "ja")))
    raw["ru"] = max(raw["ko"], raw["ja"])
    raw["en_mean"] = spread(lambda s: statistics.mean(pct(ppl(s, "c3_ru", d), base[f"ppl_{d}"]) for d in EN))
    raw["mid_mean"] = spread(lambda s: statistics.mean(pct(ppl(s, "c3_ru", l), ppl(s, "c2_ja", l)) for l in ("ko", "ja")))
    thr = {m: max(raw[m], FLOOR) for m in EN + MID + ["el", "en_mean", "mid_mean"]}
    report["thresholds"] = thr
    report["threshold_raw_2x_range"] = raw
    print("=== 흔들림 기준 (수정판, fresh 줄기 42·43 범위 x2, 최소 2.0%p): "
          + ", ".join(f"{NAMES.get(m, m)} {thr[m]:.2f}" for m in EN + MID + ["el"])
          + f" | C3 양 끝용: 영어 평균 {thr['en_mean']:.2f}, 중간 평균 {thr['mid_mean']:.2f}")
    print("    근거 (2 x 범위): " + ", ".join(f"{NAMES.get(m, m)} {raw[m]:.2f}" for m in EN)
          + f" | 그리스어 = max(ko c0→c1 {raw['el_parts']['ko']:.2f}, ja c1→c2 {raw['el_parts']['ja']:.2f}, "
            f"ru c2→c3 {raw['el_parts']['ru']:.2f}) | ko c2→c3 {raw['ko']:.2f}, ja c2→c3 {raw['ja']:.2f}, ru = max(ko, ja)")

    en = lambda ev: {d: pct(ev[f"ppl_{d}"], base[f"ppl_{d}"]) for d in EN}
    have = lambda ev, tag: ev.get(tag) is not None and all(f"ppl_{m}" in ev[tag] for m in EN + MID + ["el"])
    out = {}

    # ---- C1 ----
    print("\n=== C1 (base anchor 줄기 vs fresh 줄기, c3, 시드별 쌍)")
    if all(have(trunk[BASET[s]], "c3_ru") and have(trunk[BASET[s]], "c2_ja") and have(trunk[BASET[s]], "c1_ko")
           for s in SEEDS):
        b3 = {s: en(trunk[BASET[s]]["c3_ru"]) for s in SEEDS}
        fe = {s: en(f3[s]) for s in SEEDS}
        p1 = {d: compare({f"s{s}": b3[s][d] for s in SEEDS}, {f"s{s}": fe[s][d] for s in SEEDS}, thr[d], -1) for d in EN}
        show("C1-P1: base 줄기가 영어를 덜 잊는다 (A = base 줄기, B = fresh 줄기, 원본 대비 %)", p1)
        mid = lambda t, s, lang: pct(trunk[t]["c3_ru"][f"ppl_{lang}"], trunk[t][LEARNED_AT[lang]][f"ppl_{lang}"])
        p2 = {lang: compare({f"s{s}": mid(FRESH[s], s, lang) for s in SEEDS},
                            {f"s{s}": mid(BASET[s], s, lang) for s in SEEDS}, thr[lang], -1) for lang in LEARNED_AT}
        show("C1-P2: fresh 줄기가 한국어·일본어를 더 잘 지킨다 (A = fresh, B = base 줄기, 배운 직후 대비 c3 %)", p2)
        out["C1-P1"] = {"metrics": p1, "verdict": aggregate(p1)}
        out["C1-P2"] = {"metrics": p2, "verdict": aggregate(p2)}
        print(f"  → C1-P1 {out['C1-P1']['verdict']}, C1-P2 {out['C1-P2']['verdict']}")
    else:
        out["C1-P1"] = out["C1-P2"] = {"verdict": "결과 없음"}
        print("  결과 없음 (trunk_base_s42/s43의 c1~c3 평가가 필요)")

    # ---- C2 ----
    arms_needed = ["latest_full", "latest_ep", "base_full", "base_ep", "ft150"]
    c2_ok = all(have(branch[s], a) for s in SEEDS for a in arms_needed)
    print("\n=== C2 (M_3 분기, 줄기별 쌍: 줄기 42 = trunk_fresh_s42, 줄기 43 = trunk_fresh_s43)")
    val = {}
    if c2_ok:
        for s in SEEDS:
            m3 = f3[s]
            val[s] = {a: {"en": en(branch[s][a]),
                          "mid": {l: pct(branch[s][a][f"ppl_{l}"], m3[f"ppl_{l}"]) for l in MID},
                          "new": pct(branch[s][a]["ppl_el"], m3["ppl_el"])} for a in arms_needed}
            val[s]["M3"] = {"en": en(m3)}
        pair = lambda arm, kind, m: {f"줄기{s}": (val[s][arm][kind][m] if kind != "new" else val[s][arm]["new"])
                                     for s in SEEDS}
        # 줄기별 먼저
        print("  줄기별 값 (원본 대비 영어 %, M_3 대비 중간·새 지식 %):")
        for s in SEEDS:
            for a in arms_needed:
                v = val[s][a]
                print(f"    줄기{s} {a:<11} 영어 " + " ".join(f"{NAMES[d]} {v['en'][d]:+.1f}" for d in EN)
                      + " | 중간 " + " ".join(f"{NAMES[l]} {v['mid'][l]:+.1f}" for l in MID)
                      + f" | 그리스어 {v['new']:+.1f}")
            print(f"    줄기{s} M_3         영어 " + " ".join(f"{NAMES[d]} {val[s]['M3']['en'][d]:+.1f}" for d in EN))
        P1 = {d: compare(pair("base_full", "en", d), pair("latest_full", "en", d), thr[d], -1) for d in EN}
        show("P1: base-전체가 최신-전체보다 영어를 덜 잊는다 (A = base_full, B = latest_full)", P1)
        P2 = {l: compare(pair("latest_full", "mid", l), pair("base_full", "mid", l), thr[l], -1) for l in MID}
        show("P2: 최신-전체가 base-전체보다 중간 지식을 더 잘 지킨다 (A = latest_full, B = base_full)", P2)
        # P3: 줄기마다 (M_3 − base_full) ≥ 기준 (두 줄기 모두)
        P3 = {}
        for d in EN:
            drops = {f"줄기{s}": val[s]["M3"]["en"][d] - val[s]["base_full"]["en"][d] for s in SEEDS}
            ok = all(v >= thr[d] - EPS for v in drops.values())
            P3[d] = {"drop_from_M3": drops, "threshold": thr[d], "ok": ok}
        print("  P3 (회복): base-전체의 영어 % 변화가 M_3보다 줄기마다 기준 이상 낮다")
        for d, r in P3.items():
            print(f"    {NAMES[d]}: 감소 " + ", ".join(f"{k} {v:+.1f}" for k, v in r["drop_from_M3"].items())
                  + f" (기준 {r['threshold']:.1f}) → {'O' if r['ok'] else 'X'}")
        p3v = sum(r["ok"] for r in P3.values())
        p3_verdict = "지지" if p3v == len(EN) else "부분 지지" if p3v else "불지지"
        # P4
        src = {d: compare(pair("latest_ep", "en", d), pair("base_ep", "en", d), thr[d]) for d in EN}
        dil = {d: compare(pair("latest_full", "en", d), pair("latest_ep", "en", d), thr[d]) for d in EN}
        show("P4 출처 효과 (A = latest_ep, B = base_ep, 영어)", src)
        show("P4 희석 효과 (A = latest_full, B = latest_ep, 영어, 참고)", dil)
        p4_dom = {d: ("자기 강화" if src[d]["sign"] == 1 else "출처 효과 반대 (최신-EP가 더 좋음)" if src[d]["sign"] == -1
                      else "희석") for d in EN}
        kinds = set(p4_dom.values())
        p4_verdict = (f"{kinds.pop()} 지지 (세 영역 모두)" if len(kinds) == 1 else "혼재: " +
                      ", ".join(f"{NAMES[d]} {k}" for d, k in p4_dom.items()))
        print(f"    영역별: " + ", ".join(f"{NAMES[d]} {k}" for d, k in p4_dom.items()) + f" → {p4_verdict}")
        # FT-150 (탐색)
        ft = {}
        for other in ("latest_full", "base_full"):
            ft[other] = {**{d: compare(pair("ft150", "en", d), pair(other, "en", d), thr[d]) for d in EN},
                         **{l: compare(pair("ft150", "mid", l), pair(other, "mid", l), thr[l]) for l in MID},
                         "el": compare(pair("ft150", "new", "el"), pair(other, "new", "el"), thr["el"])}
            show(f"FT-150 비교 (탐색, 예측 없음): A = ft150, B = {other} (영어 원본 대비 %, 중간·그리스어 M_3 대비 %)", ft[other])
        out.update({"P1": {"metrics": P1, "verdict": aggregate(P1)}, "P2": {"metrics": P2, "verdict": aggregate(P2)},
                    "P3": {"metrics": P3, "verdict": p3_verdict},
                    "P4": {"source_effect": src, "dilution_effect": dil, "per_domain": p4_dom, "verdict": p4_verdict},
                    "FT150": ft, "branch_values": val})
        go = (any(r["certain"] for r in P1.values()) or any(r["certain"] for r in P2.values())
              or any(r["certain"] for r in src.values()) or any(r["certain"] for r in dil.values()))
        out["C3_go"] = {"go": go, "rule": "P1·P2의 지표 중 하나, 또는 P4의 출처·희석 효과 중 하나가 확실한 차이 (방향 무관)"}
        print(f"  → P1 {out['P1']['verdict']}, P2 {out['P2']['verdict']}, P3 {p3_verdict}, P4 {p4_verdict}")
        print(f"  → C3 진행 조건: {'진행' if go else '진행 안 함 (C3 대신 원인 분석)'}")
    else:
        for k in ("P1", "P2", "P3", "P4", "FT150"):
            out[k] = {"verdict": "결과 없음"}
        out["C3_go"] = {"go": None, "verdict": "결과 없음"}
        print("  결과 없음 (두 줄기의 C2 arm 다섯 개 평가가 필요)")

    # ---- C4: NTP(hard target) vs KL(soft target) 복습, matched-data objective comparison ----
    c4_ok = c2_ok and all(have(branch[s], a) for s in SEEDS for a in ("latest_kl", "base_kl"))
    print("\n=== C4 (같은 복습 글, 목적 함수만 KL: latest_kl vs latest_full, base_kl vs base_full, 기준점 FT-150)")
    if c4_ok:
        for s in SEEDS:
            m3 = f3[s]
            for a in ("latest_kl", "base_kl"):
                val[s][a] = {"en": en(branch[s][a]),
                             "mid": {l: pct(branch[s][a][f"ppl_{l}"], m3[f"ppl_{l}"]) for l in MID},
                             "new": pct(branch[s][a]["ppl_el"], m3["ppl_el"])}
        print("  줄기별 값 (영어 원본 대비 %):")
        for s in SEEDS:
            for a in ("latest_full", "latest_kl", "ft150", "base_full", "base_kl"):
                print(f"    줄기{s} {a:<11} " + " ".join(f"{NAMES[d]} {val[s][a]['en'][d]:+.1f}" for d in EN))
        cat_txt = {"A": "A 완전 완화: 이 설정에서 hard target 샘플링이 증폭의 주요 기여 요인",
                   "C": "C 부분 완화: hard target이 증폭에 기여하지만, 전체 분포 맞추기만으로는 없어지지 않음",
                   "B": "B 완화 없음: hard target을 전체 분포 맞추기로 바꿔도 증폭이 감지 가능하게 줄지 않음, 다른 요인 남음",
                   "A*": "범주 밖: latest_kl이 FT-150보다도 확실히 좋음 (사전 범주에 없음, 따로 보고)"}
        better = {d: compare(pair("latest_kl", "en", d), pair("latest_full", "en", d), thr[d], -1) for d in EN}
        vs_ft = {d: compare(pair("latest_kl", "en", d), pair("ft150", "en", d), thr[d]) for d in EN}
        show("C4-P1 (1) latest_kl vs latest_full (A = latest_kl, 확실히 좋아야 완화)", better)
        show("C4-P1 (2) latest_kl vs FT-150 (A = latest_kl)", vs_ft)
        cats, ratio = {}, {}
        for d in EN:
            if better[d]["outcome"] != "예측 방향 확실":
                cats[d] = "B"
            elif not vs_ft[d]["certain"]:
                cats[d] = "A"
            elif vs_ft[d]["sign"] == 1:
                cats[d] = "C"
            else:
                cats[d] = "A*"
            mf, mk, mt = (statistics.mean(val[s][a]["en"][d] for s in SEEDS) for a in ("latest_full", "latest_kl", "ft150"))
            per = {f"줄기{s}": ((val[s]["latest_full"]["en"][d] - val[s]["latest_kl"]["en"][d]) /
                               (val[s]["latest_full"]["en"][d] - val[s]["ft150"]["en"][d])
                               if val[s]["latest_full"]["en"][d] != val[s]["ft150"]["en"][d] else None) for s in SEEDS}
            ratio[d] = {"pooled": (mf - mk) / (mf - mt) if mf != mt else None, "per_trunk": per}
        kinds = set(cats.values())
        c4p1 = (cat_txt[kinds.pop()] if len(kinds) == 1 else
                "혼재: " + ", ".join(f"{NAMES[d]} {c}" for d, c in cats.items()))
        print("  영역별 범주: " + ", ".join(f"{NAMES[d]} {c}" for d, c in cats.items()) + f" → {c4p1}")
        print("  완화 비율 (latest_full − latest_kl) / (latest_full − FT-150), 서술용 (인과 해석 안 함): "
              + ", ".join(f"{NAMES[d]} " + ("-" if r["pooled"] is None else f"{r['pooled']:.2f}")
                          + " (" + ", ".join("-" if v is None else f"{k} {v:.2f}" for k, v in r["per_trunk"].items()) + ")"
                          for d, r in ratio.items()))
        p2 = {d: compare(pair("base_kl", "en", d), pair("base_full", "en", d), thr[d]) for d in EN}
        show("C4-P2 base_kl이 base_full보다 영어에서 확실히 나쁘지 않다 (A = base_kl, B = base_full)", p2)
        worse = [NAMES[d] for d, r in p2.items() if r["certain"] and r["sign"] == 1]
        c4p2 = "지지" if not worse else f"불지지 (확실히 나쁨: {', '.join(worse)})"
        print(f"  → C4-P2 {c4p2}")
        rep = {}
        for kl_arm, ntp_arm in (("latest_kl", "latest_full"), ("base_kl", "base_full")):
            rep[f"{kl_arm}_vs_{ntp_arm}"] = {
                **{l: compare(pair(kl_arm, "mid", l), pair(ntp_arm, "mid", l), thr[l]) for l in MID},
                "el": compare(pair(kl_arm, "new", "el"), pair(ntp_arm, "new", "el"), thr["el"])}
            show(f"보고만 (예측 없음): {kl_arm} vs {ntp_arm}, 중간 지식·그리스어 (M_3 대비 %)", rep[f"{kl_arm}_vs_{ntp_arm}"])
        aux = {a: {m: statistics.mean(branch[s][a][m] for s in SEEDS) for m in ("kl_base_ckpt", "param_dist")}
               for a in ("latest_full", "latest_kl", "base_full", "base_kl", "ft150")}
        print("  보조 (두 줄기 평균): " + ", ".join(f"{a} probe KL {v['kl_base_ckpt']:.4f} / 거리 {v['param_dist']:.5f}"
                                               for a, v in aux.items()))
        out["C4"] = {"P1": {"better_than_latest_full": better, "vs_ft150": vs_ft, "per_domain": cats,
                            "mitigation_ratio": ratio, "verdict": c4p1},
                     "P2": {"metrics": p2, "verdict": c4p2}, "report_only": rep, "aux": aux,
                     "values": {s: {a: val[s][a] for a in ("latest_kl", "base_kl")} for s in SEEDS}}
    else:
        out["C4"] = {"verdict": "결과 없음"}
        print("  결과 없음 (두 줄기의 C2 arm + latest_kl, base_kl 필요)")

    # ---- P5 (H3): 판정 = C2 종료 후, 줄기(FT·fresh·base) c1~c3 + C2 arm 체크포인트. C3 arm을 더한 값은 참고 ----
    def points(arms):
        pts = []
        for t, ev in trunk.items():
            for tag in ("c1_ko", "c2_ja", "c3_ru"):
                if ev.get(tag) and "kl_base_ckpt" in ev[tag] and all(f"ppl_{d}" in ev[tag] for d in EN):
                    pts.append((f"{t}/{tag}", ev[tag]))
        for s in SEEDS:
            for arm in arms:
                e = branch[s].get(arm)
                if e and "kl_base_ckpt" in e and all(f"ppl_{d}" in e for d in EN):
                    pts.append((f"branch_{s}/{arm}", e))
        return pts

    def corr(pts):
        y = [statistics.mean(en(e).values()) for _, e in pts]
        r_kl = spearman([e["kl_base_ckpt"] for _, e in pts], y)
        r_pd = spearman([e["param_dist"] for _, e in pts], y)
        per = {d: {"kl": spearman([e["kl_base_ckpt"] for _, e in pts], [en(e)[d] for _, e in pts]),
                   "param_dist": spearman([e["param_dist"] for _, e in pts], [en(e)[d] for _, e in pts])} for d in EN}
        return r_kl, r_pd, per

    print("\n=== P5 (H3): 영어 ppl % 변화(세 영역 평균)와의 Spearman 상관, probe KL vs 파라미터 거리")
    if c2_ok:
        pts = points(arms_needed)
        r_kl, r_pd, per = corr(pts)
        ok = r_kl is not None and r_pd is not None and r_kl > r_pd
        out["P5"] = {"n": len(pts), "rho_kl": r_kl, "rho_param_dist": r_pd, "per_domain": per,
                     "verdict": "지지" if ok else "불지지", "points": [n for n, _ in pts]}
        print(f"  판정 (줄기 + C2, 체크포인트 {len(pts)}개): rho(probe KL) = {r_kl:+.3f}, "
              f"rho(파라미터 거리) = {r_pd:+.3f} → {out['P5']['verdict']} (상관 비교만, 인과 주장 없음)")
        extra = [a for a in ("m1_full", "m2_full", "mix") if all(branch[s].get(a) for s in SEEDS)]
        if extra:
            pa = points(arms_needed + extra)
            a_kl, a_pd, _ = corr(pa)
            out["P5"]["with_C3_reference"] = {"n": len(pa), "rho_kl": a_kl, "rho_param_dist": a_pd}
            print(f"  참고 (C3 arm 포함, 체크포인트 {len(pa)}개): rho(probe KL) = {a_kl:+.3f}, 파라미터 거리 = {a_pd:+.3f}")
    else:
        out["P5"] = {"verdict": "결과 없음 (C2 끝난 뒤 판정)"}
        print("  결과 없음 (C2 끝난 뒤 판정)")

    # ---- C3 예측 (단조) ----
    c3_ok = c2_ok and all(have(branch[s], a) for s in SEEDS for a in ("m1_full", "m2_full"))
    print("\n=== C3 예측: anchor 나이(최신 0, M_2 1, M_1 2, base 3)에 따라 영어는 덜 잊고 중간 지식은 더 잊는 쪽으로 단조")
    print("    (수정판 기준: 두 줄기 모두 |rho| ≥ 0.8이고 예측 방향, 그리고 양 끝(나이 0 vs base)이 확실한 차이)")
    if c3_ok:
        res = {}
        arms = sorted(AGE, key=AGE.get)
        ages = [AGE[a] for a in arms]
        e_mean = {s: {a: statistics.mean(en(branch[s][a]).values()) for a in arms} for s in SEEDS}
        m_mean = {s: {a: statistics.mean(pct(branch[s][a][f"ppl_{l}"], f3[s][f"ppl_{l}"]) for l in MID) for a in arms}
                  for s in SEEDS}
        for s in SEEDS:
            e, m = [e_mean[s][a] for a in arms], [m_mean[s][a] for a in arms]
            res[f"줄기{s}"] = {"arms": arms, "english_mean": e, "mid_mean": m,
                               "rho_age_english": spearman(ages, e), "rho_age_mid": spearman(ages, m)}
            print(f"  줄기{s}: 영어 평균 " + " → ".join(f"{v:+.1f}" for v in e)
                  + f" (rho {res[f'줄기{s}']['rho_age_english']:+.2f}), 중간 평균 " + " → ".join(f"{v:+.1f}" for v in m)
                  + f" (rho {res[f'줄기{s}']['rho_age_mid']:+.2f})")
        # 양 끝: base_full(나이 3) vs latest_full(나이 0), 영어는 base가 작고 중간은 base가 크다고 예측
        ends_en = compare({f"줄기{s}": e_mean[s]["base_full"] for s in SEEDS},
                          {f"줄기{s}": e_mean[s]["latest_full"] for s in SEEDS}, thr["en_mean"], -1)
        ends_mid = compare({f"줄기{s}": m_mean[s]["base_full"] for s in SEEDS},
                           {f"줄기{s}": m_mean[s]["latest_full"] for s in SEEDS}, thr["mid_mean"], +1)
        show("양 끝 (A = base_full, B = latest_full, 평균 지표)", {"영어 평균": ends_en, "중간 평균": ends_mid})
        rho_en_ok = all((r["rho_age_english"] or 0) <= -0.8 + EPS for r in res.values())
        rho_mid_ok = all((r["rho_age_mid"] or 0) >= 0.8 - EPS for r in res.values())
        en_ok = rho_en_ok and ends_en["outcome"] == "예측 방향 확실"
        mid_ok = rho_mid_ok and ends_mid["outcome"] == "예측 방향 확실"
        verdict = ("단조 지지 (영어·중간 모두)" if en_ok and mid_ok else "부분 지지 (영어만)" if en_ok
                   else "부분 지지 (중간만)" if mid_ok else "단조 아님")
        print(f"  영어: rho 조건 {'O' if rho_en_ok else 'X'}, 양 끝 {'O' if ends_en['outcome'] == '예측 방향 확실' else 'X'} | "
              f"중간: rho 조건 {'O' if rho_mid_ok else 'X'}, 양 끝 {'O' if ends_mid['outcome'] == '예측 방향 확실' else 'X'}")
        out["C3_pred"] = {"per_trunk": res, "ends_english": ends_en, "ends_mid": ends_mid,
                          "english_ok": en_ok, "mid_ok": mid_ok, "verdict": verdict}
        if all(have(branch[s], "mix") for s in SEEDS):
            out["C3_pred"]["mix"] = {f"줄기{s}": {"english_mean": statistics.mean(en(branch[s]["mix"]).values()),
                                                 "mid_mean": statistics.mean(pct(branch[s]["mix"][f"ppl_{l}"],
                                                                                 f3[s][f"ppl_{l}"]) for l in MID)}
                                     for s in SEEDS}
            print("  혼합 50/50 (서술용): " + ", ".join(f"{k} 영어 {v['english_mean']:+.1f}, 중간 {v['mid_mean']:+.1f}"
                                                   for k, v in out["C3_pred"]["mix"].items()))
        print(f"  → {out['C3_pred']['verdict']}")
    else:
        out["C3_pred"] = {"verdict": "결과 없음"}
        print("  결과 없음 (C2 + m1_full, m2_full 필요)")

    report.update(out)
    path = args.out or os.path.join(R, "c_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
