"""
forgetting-lab / replay_quality.py

복습 데이터 품질 보고서. 여러 세트를 같은 지표로 나란히 비교한다 (예: 자기 생성 vs 위키 replay).

지표 (블록 = 512토큰):
  * 4-gram 반복률: 블록 안 토큰 4-gram 중 앞에서 이미 나온 것의 비율 (1 - 고유 4-gram 수 / 전체 4-gram 수).
  * 망가진 블록: 4-gram 반복률 0.5 이상 (절반 이상이 반복 = 같은 구절이 계속 도는 수준).
  * 내용 구성 (간단한 규칙, 대략적인 추정):
      비영어 : 글자 중 ASCII가 아닌 글자 비율 > 30%
      코드   : 코드 기호({}();=<>[]) 밀도 > 6% 또는 코드처럼 보이는 줄 비율 > 30%
      영어 산문: 영어 기능어(the, of, and ...) 비율 >= 20% 이고 글자 비율 >= 60%
      기타   : 나머지 (목록, 표, 숫자 나열, 짧은 조각 등)
  * 무작위 샘플 5개의 앞 300자 (시드 고정)
  * held-out 겹침 (오염 검사): 복습 데이터와 영어 held-out(wiki, books, web) 사이의 토큰 13-gram 겹침.
      - 겹침 블록 비율: held-out 블록 중 복습 데이터에 있는 13-gram이 하나라도 있는 블록의 비율
      - 겹침 13-gram 비율: held-out의 전체 13-gram 중 복습 데이터에도 있는 것의 비율
      - 13-gram은 블록 안에서만 만든다 (학습도 평가도 블록 단위라서). 정확 일치(해시 아님).
      - 예시: 겹치는 연속 구간이 가장 긴 사례 상위 3개

예)
  python replay_quality.py data/selfgen_replay data/en_replay --out data/selfgen_replay/quality_report.md
"""
import argparse
import json
import os
import random
import re
import statistics

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-410m"
BROKEN = 0.5
NGRAM = 13
HELDOUT = {"wiki": "data/en_heldout_wiki", "books": "data/en_heldout_books",
           "web": "data/en_heldout_web"}
STOPWORDS = set("the of and to a in is was for on that with as by it he his at from are be this "
                "an which or were has had have her she they their its not but also been i you we".split())
CODE_LINE = re.compile(r"^\s*(def |class |import |from \S+ import|#include|public |private |return\b|"
                       r"var |let |const |function\b|if \(|for \(|while \(|<\?php|</?\w+[ >])|[;{}]\s*$")


def rep4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams)


def classify(text):
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > 0.3:
        return "비영어"
    nonspace = [c for c in text if not c.isspace()]
    sym = sum(c in "{}();=<>[]" for c in nonspace) / max(1, len(nonspace))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    code_lines = sum(bool(CODE_LINE.search(ln)) for ln in lines) / max(1, len(lines))
    if sym > 0.06 or code_lines > 0.3:
        return "코드"
    words = re.findall(r"[A-Za-z]+", text.lower())
    stop = sum(w in STOPWORDS for w in words) / max(1, len(words))
    alpha = len(letters) / max(1, len(nonspace))
    if stop >= 0.20 and alpha >= 0.6:
        return "영어 산문"
    return "기타"


def ngrams(ids, n=NGRAM):
    arr = np.asarray(ids, dtype=np.int32)
    return [arr[i:i + n].tobytes() for i in range(len(arr) - n + 1)]


def gram_set(blocks):
    grams = set()
    for b in blocks:
        grams.update(ngrams(b))
    return grams


def overlapping_block_ids(replay_blocks, hblocks):
    """held-out 블록 중 복습 데이터와 겹치는 13-gram이 하나라도 있는 블록 번호 (깨끗한 부분집합 평가용)."""
    grams = gram_set(replay_blocks)
    return [i for i, hb in enumerate(hblocks) if any(g in grams for g in ngrams(hb))]


def overlap(blocks, heldouts, tok):
    """복습 블록들의 13-gram 집합과 각 held-out 세트를 비교한다."""
    grams = gram_set(blocks)
    out, cases = {}, []
    for name, hblocks in heldouts.items():
        hit_blocks = hit_grams = total = 0
        hit_ids = []
        for bi, hb in enumerate(hblocks):
            flags = [g in grams for g in ngrams(hb)]
            total += len(flags)
            hit_grams += sum(flags)
            if any(flags):
                hit_blocks += 1
                hit_ids.append(bi)
                # 겹치는 13-gram이 이어진 가장 긴 구간 = 토큰 [s, e + 13)
                best, s = (0, 0, 0), None
                for i, f in enumerate(flags + [False]):
                    if f and s is None:
                        s = i
                    elif not f and s is not None:
                        if i - s > best[0]:
                            best = (i - s, s, i - 1)
                        s = None
                n_run, s, e = best
                cases.append({"heldout": name, "block": bi, "span_tokens": n_run + NGRAM - 1,
                              "text": tok.decode(hb[s:e + NGRAM])})
        out[name] = {"blocks_with_overlap": hit_blocks, "n_blocks": len(hblocks),
                     "block_rate": hit_blocks / len(hblocks),
                     "ngram_rate": hit_grams / total, "overlap_ngrams": hit_grams,
                     "block_ids": hit_ids}
    cases.sort(key=lambda c: -c["span_tokens"])
    return out, cases[:3]


def analyze(path, tok, heldouts, n_samples=5, seed=0):
    ds = load_from_disk(path)
    blocks = [list(b) for b in ds["input_ids"]]
    ov, ov_cases = overlap(blocks, heldouts, tok) if heldouts else ({}, [])
    texts = [tok.decode(b) for b in blocks]
    reps = [rep4(b) for b in blocks]
    cats = [classify(t) for t in texts]
    order = ["영어 산문", "코드", "비영어", "기타"]
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(blocks)), min(n_samples, len(blocks))))
    q = statistics.quantiles(reps, n=10)
    return {
        "path": path, "n_blocks": len(blocks),
        "rep4_mean": statistics.mean(reps), "rep4_median": statistics.median(reps), "rep4_p90": q[-1],
        "rep4_ge_0.3": sum(r >= 0.3 for r in reps), "broken_blocks": sum(r >= BROKEN for r in reps),
        "eos_per_block": sum(b.count(tok.eos_token_id) for b in blocks) / len(blocks),
        "categories": {c: cats.count(c) for c in order},
        "samples": [{"index": i, "category": cats[i], "rep4": round(reps[i], 3),
                     "text": texts[i][:300]} for i in picks],
        "worst": [{"index": i, "rep4": round(reps[i], 3), "text": texts[i][:200]}
                  for i in sorted(range(len(reps)), key=lambda i: -reps[i])[:2]],
        "overlap": ov, "overlap_cases": ov_cases,
    }


# LAB_NOTES.md 7장 학습 진행 기준 (보고서 보기 전 확정, 2026-09-25)
MAX_BROKEN_RATE = 0.05
MIN_PROSE_RATE = 0.70
MAX_OVERLAP_BLOCKS_RATE = 0.01   # held-out별, 400블록이면 4개 이하


def judgement(names, results):
    """첫 번째 세트(후보)에 7장 진행 기준을 적용한다. 두 번째 세트(en_replay)는 참고 비교용."""
    r = results[0]
    n = r["n_blocks"]
    checks = [
        (f"망가진 블록 ≤ {MAX_BROKEN_RATE:.0%}", r["broken_blocks"] / n <= MAX_BROKEN_RATE,
         f"{r['broken_blocks']}/{n} ({r['broken_blocks'] / n:.1%})"),
        (f"영어 산문 ≥ {MIN_PROSE_RATE:.0%}", r["categories"]["영어 산문"] / n >= MIN_PROSE_RATE,
         f"{r['categories']['영어 산문']}/{n} ({r['categories']['영어 산문'] / n:.1%})"),
    ]
    for h, o in r["overlap"].items():
        checks.append((f"{h} held-out 겹침 블록 ≤ {MAX_OVERLAP_BLOCKS_RATE:.0%}",
                       o["block_rate"] <= MAX_OVERLAP_BLOCKS_RATE,
                       f"{o['blocks_with_overlap']}/{o['n_blocks']} ({o['block_rate']:.2%})"))
    ok = all(c[1] for c in checks)
    L = ["", f"## 학습 진행 판정 (LAB_NOTES 7장 기준, 대상: {names[0]})", "",
         "| 조건 | 값 | 통과 |", "|---|---|---|"]
    L += [f"| {c[0]} | {c[2]} | {'O' if c[1] else '**X**'} |" for c in checks]
    L += ["", f"**판정: {'진행 (모든 조건 통과)' if ok else '중단: 기준 위반, 학습하지 말고 원인 확인'}**"]
    if r["overlap"]:
        any_hit = {h: o["block_ids"] for h, o in r["overlap"].items() if o["block_ids"]}
        if any_hit:
            L += ["", "겹침 블록이 있음 → run04 평가 때 깨끗한 부분집합 ppl도 계산 (`clean_subset_eval.py`). "
                  "겹친 블록 번호: " + ", ".join(f"{h} {ids}" for h, ids in any_hit.items())]
        if len(results) > 1:
            ref = results[1]["overlap"]
            L += ["", f"참고 (판정에 안 씀): {names[1]}의 2배와 비교 - " + ", ".join(
                f"{h} {r['overlap'][h]['block_rate']:.2%} (2배 기준 {2 * ref[h]['block_rate']:.2%})"
                for h in r["overlap"])]
    return L


def to_markdown(results):
    names = [os.path.basename(r["path"].rstrip("/\\")) for r in results]
    L = ["# 복습 데이터 품질 보고서", "",
         f"블록 = 512토큰. 망가진 블록 = 4-gram 반복률 {BROKEN} 이상. 내용 구성은 간단한 규칙으로 낸 대략적인 추정.",
         "규칙 순서: 비영어(ASCII 아닌 글자 > 30%) → 코드(코드 기호 밀도 > 6% 또는 코드 같은 줄 > 30%) → "
         "영어 산문(기능어 ≥ 20%, 글자 ≥ 60%) → 기타. 한 블록에 문서가 여럿(eos로 구분) 섞이면 블록 전체로 한 번 분류하므로, "
         "샘플의 앞 300자와 분류가 달라 보일 수 있다.", "",
         "| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(results)]
    rows = [("블록 수", lambda r: f"{r['n_blocks']}"),
            ("4-gram 반복률 평균", lambda r: f"{r['rep4_mean']:.3f}"),
            ("4-gram 반복률 중앙값", lambda r: f"{r['rep4_median']:.3f}"),
            ("4-gram 반복률 상위 10% 경계", lambda r: f"{r['rep4_p90']:.3f}"),
            ("반복률 0.3 이상 블록", lambda r: f"{r['rep4_ge_0.3']} ({r['rep4_ge_0.3'] / r['n_blocks']:.1%})"),
            ("**망가진 블록 (≥ 0.5)**", lambda r: f"**{r['broken_blocks']} ({r['broken_blocks'] / r['n_blocks']:.1%})**"),
            ("블록당 eos (문서 경계)", lambda r: f"{r['eos_per_block']:.2f}")]
    for c in ["영어 산문", "코드", "비영어", "기타"]:
        rows.append((f"내용: {c}", lambda r, c=c: f"{r['categories'][c]} ({r['categories'][c] / r['n_blocks']:.1%})"))
    for label, f in rows:
        L.append(f"| {label} | " + " | ".join(f(r) for r in results) + " |")

    if results[0]["overlap"]:
        L += ["", f"## held-out과의 {NGRAM}-gram 겹침 (오염 검사)", "",
              "겹침 블록 비율 = held-out 블록 중 복습 데이터와 겹치는 13-gram이 하나라도 있는 블록. "
              "겹침 13-gram 비율 = held-out 전체 13-gram 중 복습 데이터에도 있는 것.", "",
              "| held-out | 지표 | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(results)]
        for h in results[0]["overlap"]:
            L.append(f"| {h} | 겹침 블록 비율 | " + " | ".join(
                f"{r['overlap'][h]['blocks_with_overlap']}/{r['overlap'][h]['n_blocks']} "
                f"({r['overlap'][h]['block_rate']:.2%})" for r in results) + " |")
            L.append(f"| {h} | 겹침 13-gram 비율 | " + " | ".join(
                f"{r['overlap'][h]['ngram_rate']:.4%}" for r in results) + " |")
        for name, r in zip(names, results):
            L += ["", f"### {name}: 겹침 사례 (연속 겹침 구간이 긴 순서, 최대 3개)", ""]
            if not r["overlap_cases"]:
                L.append("겹침 없음.")
            for c in r["overlap_cases"]:
                L += [f"**{c['heldout']} 블록 #{c['block']}**: 연속 {c['span_tokens']}토큰 겹침", "",
                      "```text", c["text"][:400], "```", ""]

    L += judgement(names, results)

    for name, r in zip(names, results):
        L += ["", f"## {name}: 무작위 샘플 5개 (앞 300자)", ""]
        for s in r["samples"]:
            L += [f"**#{s['index']}** ({s['category']}, 반복률 {s['rep4']})", "",
                  "```text", s["text"], "```", ""]
        L += [f"## {name}: 반복률 가장 높은 블록 2개 (앞 200자)", ""]
        for s in r["worst"]:
            L += [f"**#{s['index']}** (반복률 {s['rep4']})", "", "```text", s["text"], "```", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="복습 데이터 품질 보고서")
    p.add_argument("paths", nargs="+", help="load_from_disk 데이터셋 폴더들")
    p.add_argument("--out", help="보고서 .md 경로 (옆에 같은 이름 .json도 저장)")
    p.add_argument("--no_overlap", action="store_true", help="held-out 겹침 검사 생략")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    heldouts = {} if args.no_overlap else {
        name: [list(b) for b in load_from_disk(path)["input_ids"]] for name, path in HELDOUT.items()}
    results = [analyze(pth, tok, heldouts) for pth in args.paths]
    md = to_markdown(results)
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
