"""
forgetting-lab / hparams.py

논문 부록 A (하이퍼파라미터 표)를 실제 코드·설정에서 읽어 마크다운 표로 만든다. 손으로 적지 않는다.

  * 학습: cycle_train.make_targs가 만드는 TrainingArguments (옵티마이저, β, ε, weight decay, gradient clipping, LR, 정밀도, 배치),
          실제 run_config.json / train_summary.json (스텝, 블록 수).
  * 생성: 복습 폴더 source.json (temperature, top-p, top-k, 길이, 배치), gen_selfgen 상수.
  * 평가: eval_cycle 상수 (세트, probe), eval.csv의 n_blocks·dtype, build_lang_data 상수.

예)
  python hparams.py            # 표를 출력 (PAPER_DRAFT.md 부록 A에 붙임, verify_draft.py가 같은 원천으로 다시 대조)
"""
import csv
import glob
import json
import os
import re
import tempfile
import types

import build_lang_data as bld
import eval_cycle as evc
import gen_selfgen as gsg
from cycle_train import make_targs


def facts():
    tmp = tempfile.mkdtemp()
    a = types.SimpleNamespace(out=tmp, lr=1.5e-5, warmup_steps=10, logging_steps=1, seed=42)
    ntp, kl = make_targs(a, 150, False), make_targs(a, 150, True)
    cfgs = [json.load(open(p, encoding="utf-8")) for p in glob.glob("checkpoints/cycles/**/run_config.json", recursive=True)]
    sums = {os.path.dirname(p): json.load(open(p, encoding="utf-8"))
            for p in glob.glob("checkpoints/cycles/**/train_summary.json", recursive=True)}
    srcs = [json.load(open(p, encoding="utf-8")) for p in glob.glob("data/cycles/**/source.json", recursive=True)
            if "replay" in p]
    rows = [r for p in glob.glob("results/cycles/**/eval.csv", recursive=True)
            for r in csv.DictReader(open(p, encoding="utf-8")) if r["metric"].startswith("ppl_")]
    return {
        "optim": str(ntp.optim.value), "beta1": ntp.adam_beta1, "beta2": ntp.adam_beta2, "eps": ntp.adam_epsilon,
        "wd": ntp.weight_decay, "clip": ntp.max_grad_norm, "lr": ntp.learning_rate, "warmup": ntp.warmup_steps,
        "sched": str(ntp.lr_scheduler_type.value), "bf16": ntp.bf16, "gc": ntp.gradient_checkpointing,
        "ntp_micro": ntp.per_device_train_batch_size, "ntp_accum": ntp.gradient_accumulation_steps,
        "kl_micro": kl.per_device_train_batch_size, "kl_accum": kl.gradient_accumulation_steps,
        "steps_replay": max(v["steps"] for k, v in sums.items() if "trunk_FT" not in k),
        "steps_ft": max(v["steps"] for k, v in sums.items() if "trunk_FT" in k),
        "new_n": max(c["new_n"] for c in cfgs), "replay_n": max(c["n_replay_blocks"] for c in cfgs),
        "lr_cfg": {c["lr"] for c in cfgs}, "warmup_cfg": {c["warmup_steps"] for c in cfgs},
        "temperature": {s["temperature"] for s in srcs}, "top_p": {s["top_p"] for s in srcs}, "top_k": {s.get("top_k") for s in srcs},
        "gen_len": {s["block_size"] for s in srcs}, "gen_batch_B": {s["batch_size"] for s in srcs if s["seed"] // 1000 % 100 < 10},
        "gen_batch_C": {s["batch_size"] for s in srcs if s["seed"] // 1000 % 100 >= 50},
        "block": bld.BLOCK_SIZE, "eval_blocks": {int(r["n_blocks"]) for r in rows}, "eval_dtype": {r["dtype"] for r in rows},
        "probe_n": len(evc.PROBE_IDX), "probe_step": evc.PROBE_IDX[1] - evc.PROBE_IDX[0], "probe_set": evc.PROBE_SET,
        "train_cap": bld.TRAIN_PER_DOC, "held_cap": bld.HELDOUT_PER_DOC, "ngram_rule": bld.MAX_BLOCK_OVERLAP,
        "gen_start": "eos", "gsg_block": gsg.BLOCK_SIZE,
        "eval_batch": int(re.search(r'"--batch", type=int, default=(\d+)', open("eval_cycle.py", encoding="utf-8").read()).group(1)),
        "seeds": sorted({json.load(open(p, encoding="utf-8"))["seed"] for p in glob.glob("results/cycles/trunk_*/trunk_config.json")}),
    }


def table(f):
    assert f["lr_cfg"] == {f["lr"]} and f["warmup_cfg"] == {f["warmup"]}, "run_config와 코드의 LR·워밍업이 다름"
    assert len(f["temperature"]) == len(f["top_p"]) == len(f["gen_len"]) == 1
    one = lambda s: next(iter(s))
    return "\n".join([
        "| Group | Setting | Value |",
        "|---|---|---|",
        f"| Training | Optimizer | AdamW 8-bit (bitsandbytes, `{f['optim']}`), re-initialized every cycle |",
        f"| Training | Adam β1, β2 | {f['beta1']}, {f['beta2']} |",
        f"| Training | Adam ε | {f['eps']:.0e} |".replace("e-08", "e-8"),
        f"| Training | Weight decay | {f['wd']:g} |",
        f"| Training | Gradient clipping (max norm) | {f['clip']} |",
        f"| Training | Learning rate | {f['lr']:.1e}, constant after a {f['warmup']}-step linear warmup |".replace("e-05", "e-5"),
        f"| Training | Precision | fp32 weights with bf16 autocast, gradient checkpointing; weights saved in fp16 after each cycle |",
        f"| Training | Batch per update | {f['ntp_micro'] * f['ntp_accum']} blocks of {f['block']} tokens (NTP {f['ntp_micro']} × {f['ntp_accum']}, KL {f['kl_micro']} × {f['kl_accum']}) |",
        f"| Training | Steps per cycle | {f['steps_replay']} with replay ({f['new_n']:,} new + {f['replay_n']} replay blocks), {f['steps_ft']} without (FT) |",
        f"| Training | Seeds | {', '.join(map(str, f['seeds']))} (training seed = trunk seed) |",
        f"| Generation | Start and stopping | a single EOS token (unconditional); not stopped at EOS |",
        f"| Generation | Temperature | {one(f['temperature'])} |",
        f"| Generation | Top-p | {one(f['top_p'])} (no top-k; padding ids blocked) |",
        f"| Generation | Length | {one(f['gen_len'])} tokens per block |",
        f"| Generation | Batch | {one(f['gen_batch_B'])} (B stage), {one(f['gen_batch_C'])} (C stage) |",
        f"| Generation | Precision | fp16 weights, KV cache |",
        f"| Evaluation | Block length | {f['block']} tokens ({f['block'] - 1} prediction positions) |",
        f"| Evaluation | Held-out blocks | {one(f['eval_blocks'])} per set (English Wikipedia, PG-19 books, C4 web; Korean, Japanese, Russian, Greek) |",
        f"| Evaluation | Probe for probe KL | {f['probe_n']} blocks, every {f['probe_step']}th block of the English Wikipedia held-out set |",
        f"| Evaluation | Precision | {one(f['eval_dtype']).replace('float16', 'fp16')} weights, batch {f['eval_batch']} |",
        f"| Data | Article cap | {f['train_cap']} blocks (training), {f['held_cap']} blocks (held-out) |",
        f"| Data | Contamination rule | drop held-out blocks with more than {f['ngram_rule']:.0%} of their 13-grams in the training data |",
    ])


if __name__ == "__main__":
    print(table(facts()))
