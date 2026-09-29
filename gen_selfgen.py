"""
forgetting-lab / gen_selfgen.py

run04(자기 생성 복습)용 데이터: 원본 pythia-410m이 프롬프트 없이 스스로 생성한 영어 글.

  * eos 토큰 하나에서 시작해서 512토큰을 자유롭게 샘플링 (사전학습 때처럼 문서 사이가 eos로 이어지는
    분포를 따르도록, eos가 나와도 멈추지 않고 계속 생성한다).
  * 샘플링: temperature 1.0, top-p 0.95. 그 밖의 조작(top-k, 반복 벌점 등)은 없다.
    HF generate()의 기본값(top_k=50 등)이 섞이지 않게 샘플링 루프를 직접 쓴다.
  * 토크나이저에 없는 패딩 id(50277~50303)는 뽑히지 않게 막는다.
  * 블록 = 생성된 512토큰 (시작 eos는 빼고). run03 복습 데이터와 같은 2400블록.
  * 배치마다 시드(seed + 배치 번호)를 고정하고 배치 단위로 중간 저장해서, 끊겨도 이어서 돌릴 수 있다.

결과: data/selfgen_replay (+ source.json). 이미 있으면 덮어쓰지 않는다.

예)
  python gen_selfgen.py                          # 2400블록
  python gen_selfgen.py --n_blocks 64 --out <임시폴더>   # 시간 재기
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

MODEL_NAME = "EleutherAI/pythia-410m"
BLOCK_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description="원본 모델의 자기 생성 복습 데이터 만들기")
    p.add_argument("--out", default="data/selfgen_replay")
    p.add_argument("--n_blocks", type=int, default=2400)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


@torch.no_grad()
def sample_batch(model, bos, n, gen, vocab_limit, args):
    """eos 하나에서 시작해 BLOCK_SIZE 토큰을 샘플링. KV 캐시 사용."""
    cur = torch.full((n, 1), bos, dtype=torch.long, device="cuda")
    past, out = None, []
    for _ in range(BLOCK_SIZE):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :].float()
        logits[:, vocab_limit:] = float("-inf")          # 토크나이저에 없는 패딩 id
        probs = torch.softmax(logits / args.temperature, dim=-1)
        # top-p: 확률 큰 순으로 누적해서 p를 처음 넘는 지점까지만 남긴다
        sp, si = probs.sort(dim=-1, descending=True)
        drop = sp.cumsum(dim=-1) - sp > args.top_p
        sp[drop] = 0
        pick = torch.multinomial(sp / sp.sum(dim=-1, keepdim=True), 1, generator=gen)
        cur = si.gather(-1, pick)
        out.append(cur)
    return torch.cat(out, dim=1).cpu()


def main():
    args = parse_args()
    if os.path.isdir(args.out) and os.path.exists(os.path.join(args.out, "dataset_info.json")):
        raise SystemExit(f"{args.out} 가 이미 있습니다. 덮어쓰지 않습니다.")
    parts_dir = args.out.rstrip("/\\") + "_parts"
    os.makedirs(parts_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    bos = tok.eos_token_id
    vocab_limit = len(tok)

    n_batches = -(-args.n_blocks // args.batch_size)
    print(f"생성: {args.n_blocks}블록 x {BLOCK_SIZE}토큰, 배치 {args.batch_size} x {n_batches}개, "
          f"temperature {args.temperature}, top_p {args.top_p}, seed {args.seed}")
    t_start = time.time()
    done_now = 0
    for b in range(n_batches):
        part = os.path.join(parts_dir, f"part_{b:04d}.pt")
        if os.path.exists(part):
            continue   # 이어서 돌리기
        n = min(args.batch_size, args.n_blocks - b * args.batch_size)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + b)
        t0 = time.time()
        torch.save(sample_batch(model, bos, n, gen, vocab_limit, args), part)
        done_now += 1
        el = time.time() - t_start
        left = el / done_now * (n_batches - b - 1)
        print(f"  배치 {b + 1}/{n_batches}: {time.time() - t0:.0f}초 (남은 예상 {left / 60:.1f}분)")

    blocks = torch.cat([torch.load(p) for p in sorted(glob.glob(os.path.join(parts_dir, "part_*.pt")))])
    blocks = blocks[:args.n_blocks]
    assert blocks.shape == (args.n_blocks, BLOCK_SIZE), blocks.shape
    Dataset.from_dict({"input_ids": blocks.tolist()}).save_to_disk(args.out)
    eos_count = int((blocks == bos).sum())
    with open(os.path.join(args.out, "source.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": f"{MODEL_NAME} 자기 생성 (원본 모델, fp16)",
            "method": "프롬프트 없이 eos 토큰 하나에서 시작해 512토큰 샘플링. eos가 나와도 멈추지 않음 "
                      "(사전학습처럼 문서 사이가 eos로 이어짐). 블록 = 생성된 512토큰 (시작 eos 제외). "
                      "샘플링 루프 직접 구현: temperature + top-p만 적용, top-k·반복 벌점 없음, "
                      "토크나이저 밖 패딩 id 차단.",
            "temperature": args.temperature, "top_p": args.top_p, "top_k": None,
            "repetition_penalty": None, "seed": args.seed,
            "seed_rule": "배치 b는 torch.Generator(cuda).manual_seed(seed + b)",
            "batch_size": args.batch_size, "n_blocks": args.n_blocks, "block_size": BLOCK_SIZE,
            "eos_tokens_in_blocks": eos_count,
            "note": "GPU fp16 연산은 비트 단위로 재현되지 않을 수 있어, 같은 시드로 다시 돌려도 "
                    "완전히 같은 글이 나온다는 보장은 없다.",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False)
    print(f"저장: {args.out} ({len(blocks)}블록, 블록 안 eos {eos_count}개). "
          f"중간 파일 {parts_dir} 는 확인 후 지워도 됨")


if __name__ == "__main__":
    main()
