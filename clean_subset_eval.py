"""
forgetting-lab / clean_subset_eval.py

복습 데이터와 13-gram이 겹치는 held-out 블록을 뺀 "깨끗한 부분집합"으로 영어 영역별 ppl을 다시 재서,
겹침 때문에 결론이 바뀌는지 확인한다 (LAB_NOTES 7장 규칙, 발견 3 민감도 확인).

  * 겹침 블록 = replay_quality.py와 같은 정의 (토큰 13-gram 정확 일치, 블록 안에서만)
  * 평가는 eval_all.py와 같은 방식 (fp16, eval_loss, 배치 4). 전체 400블록 값도 같이 재서
    기존 결과와 같은지 확인한다.
  * --target과 --reference를 주면 "target − reference 중앙값" (원본 대비 % 변화의 차이, %p)을
    전체/깨끗한 부분집합 두 가지로 보여준다.

예)
  # run03 민감도 확인: en_replay와 겹치는 wiki 블록을 빼고
  python clean_subset_eval.py --replay data/en_replay --domains wiki ^
      --models seed43=checkpoints/run01_seed43/checkpoint-300 seed44=checkpoints/run01_seed44/checkpoint-300 ^
               run03=checkpoints/run03_replay20/checkpoint-375 ^
      --reference seed43 seed44 --target run03 --out results/fp16/clean_subset_run03.csv
"""
import argparse
import csv
import os
import statistics

import torch
from datasets import load_from_disk

from eval_korean import eval_loaded, free_cuda, load_blocks, load_model
from replay_quality import HELDOUT, overlapping_block_ids

BASE_MODEL = "EleutherAI/pythia-410m"


def main():
    p = argparse.ArgumentParser(description="겹침 블록을 뺀 held-out으로 영어 영역별 ppl 다시 재기")
    p.add_argument("--replay", required=True, help="복습 데이터 폴더 (겹침 기준)")
    p.add_argument("--domains", nargs="+", default=list(HELDOUT), choices=list(HELDOUT))
    p.add_argument("--models", nargs="+", required=True, help="이름=경로 (원본 base는 자동 추가)")
    p.add_argument("--reference", nargs="*", default=[], help="기준점 (중앙값) 모델 이름들")
    p.add_argument("--target", help="기준점과 비교할 모델 이름")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, help="테스트용: 각 세트 앞 N블록만")
    p.add_argument("--device", default="cuda", help="테스트용으로 cpu 가능 (그땐 --dtype float32)")
    p.add_argument("--out", help="결과 csv")
    args = p.parse_args()

    replay = [list(b) for b in load_from_disk(args.replay)["input_ids"]]
    sets = {}
    for dom in args.domains:
        blocks = load_blocks(HELDOUT[dom], limit=args.limit)
        drop = set(overlapping_block_ids(replay, [list(map(int, b)) for b in blocks]))
        keep = [i for i in range(len(blocks)) if i not in drop]
        sets[dom] = {"full": blocks, "clean": blocks[keep], "dropped": sorted(drop)}
        print(f"{dom}: 전체 {len(blocks)}블록, 겹침 {len(drop)}블록 제외 → 깨끗한 부분집합 {len(keep)}블록"
              + (f" (제외: {sorted(drop)})" if drop else ""))

    models = [("base", BASE_MODEL)] + [tuple(m.split("=", 1)) for m in args.models]
    dtype = getattr(torch, args.dtype)
    rows = []
    for name, path in models:
        model = load_model(path, args.device, dtype)
        for dom, s in sets.items():
            for subset in ("full", "clean"):
                loss, ppl = eval_loaded(model, s[subset], args.device, batch_size=args.batch)
                rows.append({"model": name, "path": path, "domain": dom, "subset": subset,
                             "n_blocks": len(s[subset]), "loss": loss, "ppl": ppl})
                print(f"  {name:<8} {dom:<6} {subset:<5} ({len(s[subset])}블록) ppl={ppl:.4f}")
        model = None
        free_cuda()

    ppl = {(r["model"], r["domain"], r["subset"]): r["ppl"] for r in rows}
    change = lambda m, d, s: (ppl[(m, d, s)] / ppl[("base", d, s)] - 1) * 100
    print("\n원본 대비 % 변화 (전체 → 깨끗한 부분집합)")
    for name, _ in models[1:]:
        print(f"  {name:<8} " + " | ".join(
            f"{d} {change(name, d, 'full'):+.1f}% → {change(name, d, 'clean'):+.1f}%" for d in sets))

    if args.target and args.reference:
        print(f"\n{args.target} − 기준점({', '.join(args.reference)} 중앙값), %p")
        for d in sets:
            out = []
            for s in ("full", "clean"):
                ref = statistics.median(change(m, d, s) for m in args.reference)
                out.append(f"{s} {change(args.target, d, s) - ref:+.1f}%p "
                           f"(target {change(args.target, d, s):+.1f}%, 기준점 {ref:+.1f}%)")
            print(f"  {d}: " + "  →  ".join(out))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["dropped_blocks"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "dropped_blocks": " ".join(map(str, sets[r["domain"]]["dropped"]))
                            if r["subset"] == "clean" else ""})
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
