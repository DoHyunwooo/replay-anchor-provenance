"""
forgetting-lab / gen_replay.py

임의의 체크포인트(또는 원본)에서 자기 생성 복습 데이터 N블록을 만든다 (LAB_NOTES 9장).
샘플링은 gen_selfgen.py와 같다: eos 하나에서 시작, 512토큰, temperature 1.0, top-p 0.95,
eos가 나와도 멈추지 않음, 패딩 id 차단, 배치 b의 시드 = seed + b.

  * --english_only: replay_quality.py의 분류기로 "영어 산문" 블록만 남기고, N블록이 찰 때까지 배치를 더 생성한다.
    (--max_batches를 넘으면 멈춘다. 체크포인트가 영어를 거의 안 만들 때 무한히 돌지 않게.)
  * 배치 단위 중간 저장(<out>_parts)이라 끊겨도 이어서 돈다. 설정이 다른 중간 파일이 있으면 멈춘다.
  * source.json: 생성 체크포인트와 지문, 샘플링 설정, 생성 GPU 시간(배치마다 synchronize로 잰 합),
    생성 토큰 수(버린 블록 포함), 남긴 블록 수, 전체 생성분의 내용 구성(영어 산문/코드/비영어/기타).

결과가 이미 있으면(source.json) 덮어쓰지 않는다.

생성 시드 등록부 (LAB_NOTES 9장 분기 replay 규칙): 배치 b의 시드는 seed + b라서 생성 하나가 시드 구간
[seed, seed + 최대 배치 수)를 쓴다. 새로 시작할 때 이 구간을 --seed_registry(기본 data/cycles/replay_seeds.json)에
적고, 다른 생성(다른 --out)이 이미 쓴 구간과 하나라도 겹치면 **생성하지 않고 멈춘다.** 줄기 학습에 쓰인 replay를
분기에서 다시 뽑으면 "이미 본 샘플" 효과가 나이 효과와 섞이기 때문. data/selfgen_replay(run04, seed 1234부터 50배치)도
이미 쓴 구간으로 친다. 같은 --out을 이어서 돌리는 것은 허용.

예)
  python gen_replay.py --model checkpoints/cycles/trunk_fresh_s42/c1_ko --out data/cycles/x/c2_replay --n_blocks 960
  python gen_replay.py --out <임시> --n_blocks 8 --batch_size 4 --english_only    # 테스트
"""
import argparse
import datetime
import glob
import json
import os
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_all import ckpt_fingerprint
from gen_selfgen import BLOCK_SIZE, MODEL_NAME, sample_batch
from replay_quality import classify

CATS = ["영어 산문", "코드", "비영어", "기타"]


def parse_args():
    p = argparse.ArgumentParser(description="체크포인트에서 자기 생성 복습 데이터 만들기")
    p.add_argument("--model", default=MODEL_NAME, help="체크포인트 폴더 또는 허브 이름 (기본: 원본)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_blocks", type=int, default=960)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--english_only", action="store_true", help="영어 산문 블록만 남김")
    p.add_argument("--max_batches", type=int, default=None,
                   help="english_only일 때 최대 배치 수 (기본: 필요한 배치 수 x 20)")
    p.add_argument("--seed_registry", default=REGISTRY, help="이미 쓴 생성 시드 구간 기록")
    return p.parse_args()


REGISTRY = "data/cycles/replay_seeds.json"
# 등록부 전에 만든 생성 데이터 (gen_selfgen.py): 시드 1234부터 배치 50개
LEGACY = [{"out": "data/selfgen_replay", "model": MODEL_NAME, "seed_start": 1234, "seed_end": 1284,
           "note": "run04 복습 데이터 (gen_selfgen.py, 등록부 이전)"}]


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def reserve_seeds(registry, out, model, start, end):
    """[start, end) 구간이 다른 생성과 겹치면 멈추고, 아니면 등록부에 적는다. 같은 out은 이어가기로 본다."""
    entries = []
    if os.path.exists(registry):
        with open(registry, encoding="utf-8") as f:
            entries = json.load(f)
    mine = [e for e in entries if norm(e["out"]) == norm(out)]
    if mine:
        if (mine[0]["seed_start"], mine[0]["seed_end"]) != (start, end):
            raise SystemExit(f"{out} 는 예전에 다른 시드 구간 {mine[0]['seed_start']}~{mine[0]['seed_end'] - 1}으로 "
                             f"등록돼 있습니다 ({registry}).")
        return
    clash = [e for e in LEGACY + entries if e["seed_start"] < end and start < e["seed_end"]]
    if clash:
        free = max(e["seed_end"] for e in LEGACY + entries)
        raise SystemExit("이미 쓴 생성 시드와 겹칩니다 (분기 replay 규칙, LAB_NOTES 9장): "
                         + "; ".join(f"{e['out']} ({e['model']}) 시드 {e['seed_start']}~{e['seed_end'] - 1}" for e in clash)
                         + f". 요청 구간 {start}~{end - 1}. 겹치지 않는 시드를 쓰세요 (예: --seed {free} 이상).")
    entries.append({"out": out, "model": model, "seed_start": start, "seed_end": end,
                    "registered_at": datetime.datetime.now().isoformat(timespec="seconds")})
    os.makedirs(os.path.dirname(registry) or ".", exist_ok=True)
    with open(registry, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.out, "source.json")):
        raise SystemExit(f"{args.out} 가 이미 있습니다. 덮어쓰지 않습니다.")
    parts_dir = args.out.rstrip("/\\") + "_parts"
    os.makedirs(parts_dir, exist_ok=True)

    fp = ckpt_fingerprint(args.model)
    settings = {"model": args.model, "ckpt_fp": fp, "batch_size": args.batch_size, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p}
    meta_path = os.path.join(parts_dir, "settings.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        if old != settings:
            raise SystemExit(f"{parts_dir} 의 중간 파일이 다른 설정으로 만들어졌습니다: {old} vs {settings}")
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    need = -(-args.n_blocks // args.batch_size)
    max_batches = args.max_batches or (need * 20 if args.english_only else need)
    reserve_seeds(args.seed_registry, args.out, args.model, args.seed, args.seed + max_batches)
    model = None

    kept, gpu_s, n_generated, cats_all, b = [], 0.0, 0, dict.fromkeys(CATS, 0), 0
    t_wall = time.time()
    print(f"생성: {args.model} ({fp}) → {args.out}, {args.n_blocks}블록"
          f"{' (영어 산문만)' if args.english_only else ''}, 배치 {args.batch_size}, seed {args.seed}")
    while len(kept) < args.n_blocks:
        if b >= max_batches:
            raise SystemExit(f"배치 {max_batches}개를 만들었는데 {len(kept)}/{args.n_blocks}블록뿐입니다 "
                             "(--max_batches로 늘릴 수 있음). 저장하지 않고 멈춥니다.")
        part = os.path.join(parts_dir, f"part_{b:04d}.pt")
        if os.path.exists(part):
            d = torch.load(part)
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
            gen = torch.Generator(device="cuda").manual_seed(args.seed + b)
            torch.cuda.synchronize()
            t0 = time.time()
            ids = sample_batch(model, tok.eos_token_id, args.batch_size, gen, len(tok), args)
            torch.cuda.synchronize()
            d = {"ids": ids, "gpu_s": time.time() - t0}
            torch.save(d, part)
            print(f"  배치 {b + 1}: {d['gpu_s']:.0f}초")
        gpu_s += d["gpu_s"]
        n_generated += len(d["ids"])
        for row in d["ids"]:
            c = classify(tok.decode(row))
            cats_all[c] += 1
            if (not args.english_only or c == "영어 산문") and len(kept) < args.n_blocks:
                kept.append(row)
        if args.english_only:
            print(f"    영어 산문 누적 {len(kept)}/{args.n_blocks}")
        b += 1

    blocks = torch.stack(kept)
    Dataset.from_dict({"input_ids": blocks.tolist()}).save_to_disk(args.out)
    src = {
        **settings, "english_only": args.english_only,
        "method": "eos 하나에서 시작해 512토큰 샘플링, eos에서 멈추지 않음. temperature + top-p만 적용, "
                  "패딩 id 차단 (gen_selfgen.py와 같은 샘플링 루프). 배치 b의 시드 = seed + b."
                  + (" replay_quality.classify가 '영어 산문'인 블록만 순서대로 남김." if args.english_only else ""),
        "n_blocks": len(blocks), "block_size": BLOCK_SIZE,
        "n_batches": b, "generated_blocks": n_generated,
        "generated_tokens": n_generated * BLOCK_SIZE, "kept_rate": len(blocks) / n_generated,
        "categories_all_generated": cats_all,
        "gen_gpu_seconds": round(gpu_s, 1),
        "wall_seconds_this_run": round(time.time() - t_wall, 1),
        "eos_tokens_in_blocks": int((blocks == tok.eos_token_id).sum()),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, "source.json"), "w", encoding="utf-8") as f:
        json.dump(src, f, indent=2, ensure_ascii=False)
    print(f"저장: {args.out} ({len(blocks)}블록, 생성 {n_generated}블록, GPU {gpu_s:.0f}초, "
          f"구성 {cats_all})")


if __name__ == "__main__":
    main()
