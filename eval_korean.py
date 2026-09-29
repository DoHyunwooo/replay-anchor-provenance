"""
forgetting-lab / eval_korean.py

"영어를 잊는 동안 한국어는 얼마나 늘었나"를 재는 쪽 절반.
원본 pythia-410m과 체크포인트 6개를 data/ko_heldout(학습에 안 쓴 한국어 400블록)에
돌려서 평균 loss와 perplexity를 구하고 results/ko_perplexity.csv로 저장한다.
"""
import gc
import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

# ===== 설정 =====
BASE_MODEL = "EleutherAI/pythia-410m"
CKPT_DIR = "checkpoints/run01_ko_wiki"
STEPS = [50, 100, 150, 200, 250, 300]
HELDOUT = "data/ko_heldout"
OUT_CSV = "results/ko_perplexity.csv"
BATCH_SIZE = 4          # OOM 나면 자동으로 절반씩 줄임
# ===============


def load_blocks(path=HELDOUT, limit=None):
    """held-out 블록을 (N, BLOCK) 텐서로 읽는다. limit을 주면 앞에서 N개만."""
    blocks = load_from_disk(path).with_format("torch")["input_ids"][:]
    return blocks if limit is None else blocks[:limit]


def load_model(model_path, device, dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    return model.to(device).eval()


def free_cuda():
    """호출한 쪽에서 모델 참조를 del 한 다음 부른다."""
    gc.collect()
    torch.cuda.empty_cache()


def eval_loss(model_path, blocks, device, batch_size=BATCH_SIZE, dtype=torch.float32):
    """블록들의 토큰 단위 평균 NLL을 구한다. 반환: (평균 loss, perplexity)

    dtype은 모델 가중치 정밀도. 기본 fp32는 8GB에 410m가 충분히 들어가고 loss를
    소수점까지 믿을 수 있다. lm-eval을 fp16으로 돌릴 때는 조건을 맞추려고
    float16을 넘긴다 (cross-entropy 자체는 어느 쪽이든 fp32로 계산).
    """
    model = load_model(model_path, device, dtype)
    result = eval_loaded(model, blocks, device, batch_size)
    del model
    free_cuda()
    return result


@torch.no_grad()
def eval_loaded(model, blocks, device, batch_size=BATCH_SIZE):
    """이미 올려둔 모델로 eval_loss와 똑같이 계산한다 (여러 데이터셋을 한 번 로드로 잴 때)."""
    total_nll, total_tokens = 0.0, 0
    i = 0
    while i < len(blocks):
        bs = batch_size
        while True:
            try:
                ids = blocks[i : i + bs].to(device)
                logits = model(input_ids=ids).logits
                # 다음 토큰 예측이므로 한 칸씩 밀어서 비교 (블록당 511개 예측)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                shift_labels = ids[:, 1:].reshape(-1)
                nll = F.cross_entropy(
                    shift_logits.float(), shift_labels, reduction="sum"
                )
                total_nll += nll.item()
                total_tokens += shift_labels.numel()
                del ids, logits, shift_logits, shift_labels, nll
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"    (OOM -> 배치 {bs}로 줄여서 재시도)")
                if bs == 1:
                    batch_size = 1
        i += bs

    mean_loss = total_nll / total_tokens
    return mean_loss, math.exp(mean_loss)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = load_blocks()            # (400, 512) 텐서로 한 번에 올려둠 (작아서 괜찮음)
    print(f"검증 데이터: {tuple(blocks.shape)} (블록 x 토큰), device={device}\n")

    targets = [("base", 0, BASE_MODEL)]
    targets += [(f"ckpt-{s}", s, os.path.join(CKPT_DIR, f"checkpoint-{s}")) for s in STEPS]

    rows = []
    for name, step, path in targets:
        print(f"[{name}] {path} 평가 중...")
        loss, ppl = eval_loss(path, blocks, device)
        print(f"    loss={loss:.4f}  ppl={ppl:.2f}")
        rows.append({"model": name, "step": step, "ko_loss": loss, "ko_ppl": ppl})

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n=== 한국어 held-out 결과 (블록 {len(blocks)}개) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
