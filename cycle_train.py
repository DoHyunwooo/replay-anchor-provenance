"""
forgetting-lab / cycle_train.py

다회차 연구(LAB_NOTES 9장)의 cycle 하나를 학습한다.

  * 시작점: --init (체크포인트 폴더, 또는 base = 원본 pythia-410m). fp32로 올려 bf16 혼합정밀도로 학습.
  * 옵티마이저는 cycle마다 새로 만든다 (가중치만 이어받음). adamw 8bit, 배치 4 x accum 8 (스텝당 32블록).
  * 학습률: 10스텝 선형 워밍업 후 1.5e-5 고정 (constant_with_warmup).
  * 데이터: 새 언어 앞 --new_n블록 (기본 3840) + 복습 --replay_n블록 (기본 960). 합쳐서 Trainer가 seed로 셔플.
    스텝 수 = 전체 블록 / 32 (복습 있으면 4800/32 = 150, 없으면 3840/32 = 120).
  * 복습 혼합: --replay에 폴더 두 개를 주면 각 폴더 앞에서부터 replay_n/2블록씩 (50/50).
  * --new_repeat N (C단계 FT-150 통제): 복습 대신 새 데이터 중 무작위 N블록(random.Random(seed)로 고정)을 한 번 더
    넣는다. 복습과 같은 자리(새 데이터 뒤)에 붙으므로 전체 블록 수·셔플 순서·"복습 슬롯" 위치가 복습 arm과 같다.
    --replay와 같이 쓸 수 없다.
  * --replay_loss kl (C단계 sanity용): 복습 블록에서는 NTP 대신 생성 체크포인트(teacher) 대비 토큰 단위
    forward KL(teacher || 학생)을 쓴다. teacher는 복습 폴더 source.json의 model을 fp16으로 GPU에 함께 올린다.
    새 언어 블록은 그대로 NTP. loss = (새 블록 CE 합 + 복습 블록 KL 합) / 전체 예측 토큰 수.
  * 끝에서 가중치만 fp16으로 저장 (중간 저장 없음). logging_steps 1 → train_log.jsonl,
    설정 → run_config.json, 끝나면 train_summary.json (학습 시간, 토큰 수, GPU 최대 메모리).
    train_summary.json이 있으면 이미 끝난 cycle로 보고 다시 학습하지 않는다.

예)
  python cycle_train.py --init base --new_data data/cycles/ko_train --out checkpoints/cycles/x/c1_ko --seed 42
  python cycle_train.py --init base --new_data data/cycles/ko_train --replay data/cycles/x/c1_replay --out ... --seed 42
  python cycle_train.py ... --new_n 64 --replay_n 16 --max_steps 3      # 테스트
"""
import argparse
import datetime
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          default_data_collator)

from train import BATCH_SIZE, GRAD_ACCUM, MODEL_NAME, JsonlLogCallback, cast_state_dict

LR, WARMUP = 1.5e-5, 10


def parse_args():
    p = argparse.ArgumentParser(description="다회차 연구: cycle 하나 학습")
    p.add_argument("--init", required=True, help="시작 체크포인트 폴더, 또는 base")
    p.add_argument("--new_data", required=True, help="새 언어 학습 블록 폴더")
    p.add_argument("--new_n", type=int, default=3840)
    p.add_argument("--replay", nargs="*", default=[], help="복습 폴더 0~2개 (2개면 50/50)")
    p.add_argument("--replay_n", type=int, default=960)
    p.add_argument("--replay_loss", choices=["ntp", "kl"], default="ntp")
    p.add_argument("--new_repeat", type=int, default=0, help="복습 대신 새 데이터 무작위 N블록 반복 (FT-150)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--warmup_steps", type=int, default=WARMUP)
    p.add_argument("--max_steps", type=int, default=None, help="기본: 전체 블록 / 32 (테스트용으로만 바꿈)")
    p.add_argument("--logging_steps", type=int, default=1)
    args = p.parse_args()
    if len(args.replay) > 2:
        p.error("--replay는 폴더 2개까지")
    if len(args.replay) == 2 and args.replay_n % 2:
        p.error("복습 폴더 2개를 섞을 땐 --replay_n이 짝수여야 합니다")
    if args.replay_loss == "kl" and len(args.replay) != 1:
        p.error("--replay_loss kl은 복습 폴더 1개일 때만 (teacher가 하나여야 함)")
    if args.new_repeat and args.replay:
        p.error("--new_repeat는 --replay와 같이 쓸 수 없습니다")
    if not 0 <= args.new_repeat <= args.new_n:
        p.error("--new_repeat는 0 이상 new_n 이하")
    return args


def take(path, n):
    ds = load_from_disk(path)
    if len(ds) < n:
        raise SystemExit(f"{path} 에 블록이 {len(ds)}개뿐인데 {n}개가 필요합니다.")
    return ds.select(range(n)).select_columns(["input_ids"])


def replay_teacher(path):
    with open(os.path.join(path, "source.json"), encoding="utf-8") as f:
        return json.load(f)["model"]


ARCH_KEYS = ["architectures", "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
             "intermediate_size", "rotary_pct", "rotary_emb_base", "max_position_embeddings", "use_parallel_residual"]


def verify_teacher(replay_dir, init):
    """KL teacher 고정 (C4): 복습 폴더 source.json의 생성 체크포인트와 지문이 같은지, 토크나이저·구조가 학생과 같은지.
    하나라도 다르면 학습하지 않고 멈춘다. 반환: run_config에 남길 기록."""
    from transformers import AutoConfig
    from eval_all import ckpt_fingerprint
    with open(os.path.join(replay_dir, "source.json"), encoding="utf-8") as f:
        src = json.load(f)
    teacher, recorded = src["model"], src.get("ckpt_fp")
    actual = ckpt_fingerprint(teacher)
    if recorded != actual:
        raise SystemExit(f"KL teacher 지문 불일치: 복습 생성 때 {recorded}, 지금 {teacher} = {actual}. 멈춤.")
    vocab_same = AutoTokenizer.from_pretrained(teacher).get_vocab() == AutoTokenizer.from_pretrained(MODEL_NAME).get_vocab()
    ct, cs = AutoConfig.from_pretrained(teacher).to_dict(), AutoConfig.from_pretrained(init).to_dict()
    arch_diff = {k: (ct.get(k), cs.get(k)) for k in ARCH_KEYS if ct.get(k) != cs.get(k)}
    if not vocab_same or arch_diff:
        raise SystemExit(f"KL teacher와 학생이 다름: 어휘 같음 {vocab_same}, 구조 차이 {arch_diff}. 멈춤.")
    return {"teacher": teacher, "teacher_fp_at_generation": recorded, "teacher_fp_now": actual,
            "teacher_fp_match": True, "tokenizer_vocab_same": True, "architecture_same": True,
            "replay_source_seed": src.get("seed"), "replay_source_created_at": src.get("created_at")}


class KLReplayTrainer(Trainer):
    """복습 블록은 teacher 대비 forward KL, 새 블록은 NTP. 학습 과정의 나머지는 Trainer와 같다.

    위치와 평균 (NTP arm과 같게): 둘 다 다음 토큰 예측 위치(블록마다 0~510번 위치가 1~511번 토큰을 예측, 511개)에서 계산.
    블록은 모두 512토큰으로 꽉 차 있어 패딩·마스크(-100)가 없다. KL은 위치마다 전체 어휘에 대해 합한 뒤, NTP와 같은
    위치 집합의 개수로 나눈다. micro-batch마다 (CE 합 + KL 합) / (블록 수 x 511)을 내고 Trainer가 모으기 횟수(16)로 나누므로
    스텝 전체로는 "32블록 x 511위치 평균" = NTP arm(HF 손실: 합 / labels[..., 1:] 개수)과 같은 기준이다.
    기록: 스텝마다 새 데이터 CE, 복습 KL, 복습 CE(학생이 복습 토큰에 대해 낸 NTP 손실, 참고)를 train_log에 따로 남긴다."""

    def __init__(self, *a, teacher, **kw):
        super().__init__(*a, **kw)
        self.teacher = teacher
        # loss를 직접 평균 내므로 Trainer가 gradient accumulation 횟수로 나누게 한다
        self.model_accepts_loss_kwargs = False
        self._acc = {"new_ce": 0.0, "new_pos": 0, "rep_kl": 0.0, "rep_ce": 0.0, "rep_pos": 0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_rep = inputs.pop("is_replay").bool()
        labels = inputs.pop("labels")
        ids = inputs["input_ids"]
        out = model(input_ids=ids)
        logits = out.logits[:, :-1].float()
        tgt = labels[:, 1:]
        total = 0.0
        if (~is_rep).any():
            ce = F.cross_entropy(logits[~is_rep].reshape(-1, logits.size(-1)), tgt[~is_rep].reshape(-1), reduction="sum")
            total = total + ce
            self._acc["new_ce"] += ce.item()
            self._acc["new_pos"] += tgt[~is_rep].numel()
        if is_rep.any():
            with torch.no_grad():
                t_logp = F.log_softmax(self.teacher(input_ids=ids[is_rep]).logits[:, :-1].float(), dim=-1)
            s_logp = F.log_softmax(logits[is_rep], dim=-1)
            kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="sum")
            total = total + kl
            self._acc["rep_kl"] += kl.item()
            self._acc["rep_ce"] += -s_logp.gather(-1, tgt[is_rep].unsqueeze(-1)).sum().item()
            self._acc["rep_pos"] += tgt[is_rep].numel()
        loss = total / tgt.numel()
        return (loss, out) if return_outputs else loss

    def log(self, logs, *a, **kw):
        a_ = self._acc
        if "loss" in logs and a_["new_pos"] + a_["rep_pos"]:
            if a_["new_pos"]:
                logs["new_ce"] = a_["new_ce"] / a_["new_pos"]
            if a_["rep_pos"]:
                logs["replay_kl"] = a_["rep_kl"] / a_["rep_pos"]
                logs["replay_ce"] = a_["rep_ce"] / a_["rep_pos"]
            logs["replay_pos_frac"] = a_["rep_pos"] / (a_["new_pos"] + a_["rep_pos"])
            self._acc = {k: 0.0 if isinstance(v, float) else 0 for k, v in a_.items()}
        super().log(logs, *a, **kw)


def build_dataset(args):
    """새 데이터 + 복습(또는 새 데이터 반복)을 합친 학습 세트. 반환: ds, 복습 블록 수, 반복 인덱스."""
    new = take(args.new_data, args.new_n)
    parts = [new]
    if args.replay:
        per = args.replay_n // len(args.replay)
        parts += [take(r, per).cast(new.features) for r in args.replay]
    repeat_idx = []
    if args.new_repeat:
        repeat_idx = sorted(random.Random(args.seed).sample(range(args.new_n), args.new_repeat))
        parts.append(new.select(repeat_idx))
    ds = concatenate_datasets(parts)
    kl = args.replay_loss == "kl"
    ds = ds.map(lambda b, idx: {"labels": b["input_ids"],
                                **({"is_replay": [int(i >= args.new_n) for i in idx]} if kl else {})},
                batched=True, with_indices=True)
    return ds, len(ds) - args.new_n - args.new_repeat, repeat_idx


def make_targs(args, max_steps, kl):
    # kl은 teacher와 전체 어휘 분포(fp32)를 여러 개 들고 있어서 8GB에 빠듯하다 (배치 4면 최대 7.45GiB).
    # 한 번에 2블록 x 16번 모으기로 바꾼다. 스텝당 32블록은 같다.
    micro, accum = (2, 16) if kl else (BATCH_SIZE, GRAD_ACCUM)
    return TrainingArguments(
        output_dir=args.out, max_steps=max_steps,
        per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=args.warmup_steps,
        bf16=True, gradient_checkpointing=True, optim="adamw_bnb_8bit",
        logging_steps=args.logging_steps, save_strategy="no", report_to="none",
        seed=args.seed, dataloader_num_workers=0,
        remove_unused_columns=not kl,   # kl이면 is_replay 컬럼을 compute_loss까지 넘긴다
    )


def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.out, "train_summary.json")):
        raise SystemExit(f"{args.out} 는 이미 학습이 끝났습니다 (train_summary.json). 다시 하지 않습니다.")
    os.makedirs(args.out, exist_ok=True)
    init = MODEL_NAME if args.init == "base" else args.init
    kl = args.replay_loss == "kl"
    teacher_check = verify_teacher(args.replay[0], init) if kl else None

    ds, n_rep, repeat_idx = build_dataset(args)
    epoch_steps = len(ds) // (BATCH_SIZE * GRAD_ACCUM)
    max_steps = args.max_steps or epoch_steps
    print(f"[cycle] {init} → {args.out} | 새 {args.new_n} ({args.new_data}) + 복습 {n_rep} "
          f"({', '.join(args.replay) or '없음'}, loss {args.replay_loss})"
          + (f" + 새 데이터 반복 {args.new_repeat}" if args.new_repeat else "") + f" = {len(ds)}블록, "
          f"{max_steps}스텝, lr {args.lr} (워밍업 {args.warmup_steps} 후 고정), seed {args.seed}")

    config = {**vars(args), "init_resolved": init, "max_steps": max_steps, "epoch_steps": epoch_steps,
              "n_new_blocks": args.new_n, "n_replay_blocks": n_rep,
              "n_repeat_blocks": args.new_repeat, "repeat_indices_first10": repeat_idx[:10],
              "batch_size": BATCH_SIZE,
              "grad_accum": GRAD_ACCUM, "lr_schedule": "constant_with_warmup",
              "optimizer": "adamw_bnb_8bit (cycle마다 새로)", "save_dtype": "float16 (끝에서 한 번)"}
    if kl:
        config.update({"kl_teacher": teacher_check["teacher"], "kl_teacher_check": teacher_check,
                       "micro_batch": 2, "grad_accum": 16,
                       "comparison": "matched-data objective comparison: 짝이 되는 NTP arm과 같은 데이터·순서·스텝·LR·시드, "
                                     "목적 함수만 다름 (복습 블록: NTP 대신 teacher 대비 위치별 전체 어휘 forward KL)"})
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(init, dtype=torch.float32)
    targs = make_targs(args, max_steps, kl)
    common = dict(model=model, args=targs, train_dataset=ds, data_collator=default_data_collator,
                  processing_class=tok,
                  callbacks=[JsonlLogCallback(os.path.join(args.out, "train_log.jsonl"))])
    if kl:
        teacher = AutoModelForCausalLM.from_pretrained(config["kl_teacher"], dtype=torch.float16).to("cuda").eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        trainer = KLReplayTrainer(**common, teacher=teacher)
    else:
        trainer = Trainer(**common)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    # 가중치만 fp16으로 (메모리의 학습 모델은 그대로 두고 CPU 복사본만 변환)
    model.save_pretrained(args.out, state_dict=cast_state_dict(model.state_dict(), torch.float16))
    cfg_path = os.path.join(args.out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["dtype"] = "float16"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    tok.save_pretrained(args.out)

    summary = {"train_seconds": round(train_s, 1), "steps": max_steps,
               "train_tokens": max_steps * BATCH_SIZE * GRAD_ACCUM * 512,
               "new_tokens": args.new_n * 512, "replay_tokens": n_rep * 512,
               "repeat_tokens": args.new_repeat * 512,
               "full_epoch": max_steps == epoch_steps,
               "peak_gpu_gib": round(peak, 2),
               "finished_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.out, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"완료: {args.out} ({train_s:.0f}초, GPU 최대 {peak:.2f}GiB)")


if __name__ == "__main__":
    main()
