"""
forgetting-lab / measure_kl_floor.py

KL 목적의 바닥값 (LAB_NOTES 9장, 논문 §6): 학생과 교사가 **같은 가중치**일 때, 학습 때와 같은 조건
(학생 = fp32 가중치 + bf16 autocast forward, 교사 = fp16)에서 복습 글의 토큰 단위 KL(교사 ‖ 학생)의 평균.
학습 첫 스텝의 복습 KL에 해당한다 (첫 스텝 LR은 0이라 그 전에 가중치가 바뀌지 않음).

  * latest 교사: 두 fresh 줄기의 M_3 (c3_ru), 글 = 그 줄기의 C2 latest_full 복습 960블록
  * base 교사  : 원본 pythia-410m, 글 = 그 줄기의 C2 base_full 복습 960블록
  * 위치·평균은 cycle_train.KLReplayTrainer와 같다: 블록마다 0~510번 위치, 위치마다 전체 어휘 합, 위치 수로 평균.
  * 평가만 한다 (학습 없음). 결과: results/cycles/kl_floor.json

예)
  python measure_kl_floor.py
  python measure_kl_floor.py --limit 16      # 빠른 점검
"""
import argparse
import datetime
import json
import os
import statistics

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

BASE = "EleutherAI/pythia-410m"
SEEDS = [42, 43]


@torch.no_grad()
def kl_floor(model_path, replay_dir, limit=None, batch=4):
    student = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).cuda().train()
    teacher = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).cuda().eval()
    blocks = torch.tensor(load_from_disk(replay_dir)["input_ids"])[:limit]
    per_block = []
    for i in range(0, len(blocks), batch):
        ids = blocks[i:i + batch].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = student(input_ids=ids).logits[:, :-1]
        s_logp = F.log_softmax(s_logits.float(), -1)
        t_logp = F.log_softmax(teacher(input_ids=ids).logits[:, :-1].float(), -1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)          # (B, 511): 위치마다 어휘 합
        per_block += kl.mean(-1).tolist()
    del student, teacher
    torch.cuda.empty_cache()
    return {"teacher_and_student": model_path, "replay": replay_dir, "n_blocks": len(per_block),
            "positions_per_block": 511, "mean_kl_nats_per_token": statistics.mean(per_block),
            "sd_across_blocks": statistics.stdev(per_block) if len(per_block) > 1 else None,
            "min_block": min(per_block), "max_block": max(per_block)}


def main():
    p = argparse.ArgumentParser(description="KL 바닥값 재측정 (학생 = 교사, bf16 학습 forward)")
    p.add_argument("--limit", type=int, help="블록 수 제한 (점검용)")
    p.add_argument("--out", default="results/cycles/kl_floor.json")
    args = p.parse_args()
    res = {"what": "KL(교사 fp16 ‖ 학생 fp32+bf16 autocast), 학생과 교사가 같은 가중치. 학습 첫 스텝의 복습 KL과 같은 조건.",
           "created_at": datetime.datetime.now().isoformat(timespec="seconds"), "limit": args.limit, "sets": {}}
    for s in SEEDS:
        for key, model, arm in (("latest", f"checkpoints/cycles/trunk_fresh_s{s}/c3_ru", "latest_full"),
                                ("base", BASE, "base_full")):
            r = kl_floor(model, f"data/cycles/branches/trunk_fresh_s{s}/{arm}_replay", args.limit)
            res["sets"][f"{key}_s{s}"] = r
            print(f"{key} 교사, 줄기 {s}: 평균 {r['mean_kl_nats_per_token']:.5f} nats/토큰 "
                  f"(블록 {r['n_blocks']}개, 블록 간 sd {r['sd_across_blocks']:.5f}, 범위 {r['min_block']:.5f}~{r['max_block']:.5f})")
    for key in ("latest", "base"):
        res[f"{key}_mean"] = statistics.mean(res["sets"][f"{key}_s{s}"]["mean_kl_nats_per_token"] for s in SEEDS)
    print(f"latest 교사 평균 {res['latest_mean']:.5f}, base 교사 평균 {res['base_mean']:.5f} nats/토큰")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
