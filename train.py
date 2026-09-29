"""
forgetting-lab / train.py

Pythia-410m에 한국어 위키백과를 학습시키면서 일정 스텝마다 체크포인트를 저장한다.
나중에 각 체크포인트를 lm-eval로 다시 재서 "영어 능력을 얼마나 잊었나" 곡선을 그린다.

런끼리 비교하려면 데이터와 데이터 순서가 완전히 같아야 하므로:
  * 블록은 한 번만 만들어 data/ko_train에 캐시하고, 이후 런은 그걸 그대로 쓴다.
  * 데이터 크기는 DATA_STEPS(고정)로 정한다. --max_steps는 "몇 스텝에서 멈출지"만
    바꾸므로, 짧은 테스트 실행을 해도 데이터셋과 셔플 순서가 달라지지 않는다.
  * 셔플 순서는 Trainer가 --seed로 정한다. seed가 같으면 순서도 같다.

영어 복습(replay): --replay_data를 주면 한국어 블록 전체에 영어 위키 블록을 더해
섞는다. --replay_ratio는 "전체 중 영어 비율"이라 0.2면 한국어 9600 + 영어 2400.
스텝 수는 데이터 한 바퀴(블록 수 / 32)로 정해지므로 replay 런은 375스텝이 된다.
replay 없이 돌리면(기본) 데이터·순서·스텝 수 모두 예전 런과 똑같다.

체크포인트 저장 형식: 학습은 fp32 가중치 + bf16 혼합정밀도 그대로 하고, 저장하는 파일만
fp16으로 바꾼다 (--save_dtype, 기본 float16, 파일 크기 절반). 메모리의 학습 모델은
건드리지 않고 CPU 복사본만 변환한다. run01~03은 fp32로 저장된 런이다.

예)
  python train.py                                          # run01 설정 그대로
  python train.py --run_name run02_warmup100 --warmup_steps 100 --save_every 25
  python train.py --run_name run03_replay20 --save_every 25 \\
                  --replay_data data/en_replay --replay_ratio 0.2
"""
import argparse
import json
import os

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

# ===== 실험 설정 (런끼리 고정되어야 하는 값들) =====
MODEL_NAME = "EleutherAI/pythia-410m"
BLOCK_SIZE = 512        # 한 번에 보는 토큰 길이
BATCH_SIZE = 4          # GPU에 한 번에 올리는 블록 수
GRAD_ACCUM = 8          # 8번 모아서 한 번 업데이트 -> 스텝당 4*8*512 = 16,384 토큰
LEARNING_RATE = 3e-5
DATA_STEPS = 300        # 데이터셋 크기를 정하는 기준 스텝 수 (절대 바꾸지 말 것)
HELDOUT_BLOCKS = 400    # 학습에 안 쓰고 남겨두는 한국어 블록 ("배웠나" 측정용)
TRAIN_CACHE = "data/ko_train"
HELDOUT_DIR = "data/ko_heldout"
# ================================================


def parse_args():
    p = argparse.ArgumentParser(description="한국어 위키로 pythia-410m 이어학습")
    p.add_argument("--run_name", default="run01_ko_wiki",
                   help="체크포인트가 checkpoints/<run_name> 에 저장된다")
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="데이터 셔플 순서까지 결정한다. 런 비교 시에는 그대로 둘 것")
    p.add_argument("--max_steps", type=int, default=None,
                   help="학습을 멈출 스텝 수 (기본: 데이터 한 바퀴 = 블록 수/32). "
                        "데이터셋 크기에는 영향 없음 (테스트용)")
    p.add_argument("--replay_data", default=None,
                   help="영어 복습 블록 캐시 경로 (예: data/en_replay). 없으면 새로 만든다")
    p.add_argument("--replay_ratio", type=float, default=0.0,
                   help="전체 학습 블록 중 영어 비율 (0.2 = 한국어 9600 + 영어 2400)")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16",
                   help="체크포인트 파일 정밀도 (학습 정밀도와 무관). run01~03은 float32")
    p.add_argument("--logging_steps", type=int, default=10,
                   help="loss·grad_norm 기록 간격. 1이면 매 스텝 (run01~03은 10). "
                        "기록은 checkpoints/<run>/train_log.jsonl 에도 남는다")
    args = p.parse_args()
    if (args.replay_data is None) != (args.replay_ratio == 0):
        p.error("--replay_data와 --replay_ratio는 같이 주세요")
    if not 0 <= args.replay_ratio < 1:
        p.error("--replay_ratio는 0 이상 1 미만")
    return args


def build_wiki_blocks(tokenizer, n_blocks_needed, config="20231101.ko"):
    """위키백과를 스트리밍으로 읽어 BLOCK_SIZE 토큰 블록으로 자른다 (전체 다운로드 X)."""
    stream = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    buffer, blocks = [], []
    next_report = 1000
    for article in stream:
        ids = tokenizer(article["text"], verbose=False)["input_ids"]
        buffer.extend(ids + [tokenizer.eos_token_id])
        while len(buffer) >= BLOCK_SIZE:
            blocks.append(buffer[:BLOCK_SIZE])
            buffer = buffer[BLOCK_SIZE:]
        if len(blocks) >= next_report:
            print(f"  데이터 준비 중... {len(blocks)}/{n_blocks_needed} 블록")
            next_report += 1000
        if len(blocks) >= n_blocks_needed:
            break
    return blocks[:n_blocks_needed]


def prepare_train_dataset(tokenizer):
    """학습용 블록을 캐시에서 읽거나, 없으면 만들어서 캐시에 저장한다."""
    n_train = DATA_STEPS * BATCH_SIZE * GRAD_ACCUM

    if os.path.isdir(TRAIN_CACHE):
        ds = load_from_disk(TRAIN_CACHE)
        if len(ds) != n_train:
            raise SystemExit(
                f"{TRAIN_CACHE} 의 블록 수가 {len(ds)}개인데 {n_train}개가 필요합니다. "
                "설정이 바뀌었다면 캐시를 지우고 다시 만드세요."
            )
        print(f"학습 블록 {len(ds)}개를 캐시에서 불러옴 ({TRAIN_CACHE})")
    else:
        print(f"학습 토큰 수: 약 {n_train * BLOCK_SIZE / 1e6:.1f}M")
        blocks = build_wiki_blocks(tokenizer, n_train + HELDOUT_BLOCKS)
        heldout, train = blocks[-HELDOUT_BLOCKS:], blocks[:-HELDOUT_BLOCKS]

        # 맨 뒤 블록들은 학습에 안 쓰고 따로 저장 (나중에 한국어 실력 측정용).
        # 이미 있으면 절대 덮어쓰지 않는다 - 덮어쓰면 기존 평가 결과와 비교가 깨진다.
        if os.path.isdir(HELDOUT_DIR):
            saved = load_from_disk(HELDOUT_DIR)
            same = (len(saved) == len(heldout)
                    and list(saved[0]["input_ids"]) == heldout[0]
                    and list(saved[-1]["input_ids"]) == heldout[-1])
            if not same:
                raise SystemExit(
                    f"새로 만든 held-out이 기존 {HELDOUT_DIR} 과 다릅니다. "
                    "데이터 소스가 바뀌었을 수 있어 런 비교가 불가능하니 중단합니다."
                )
            print(f"기존 held-out {len(saved)}개와 일치 확인")
        else:
            Dataset.from_dict({"input_ids": heldout}).save_to_disk(HELDOUT_DIR)
            print(f"held-out {len(heldout)}개 저장 ({HELDOUT_DIR})")

        ds = Dataset.from_dict({"input_ids": train})
        ds.save_to_disk(TRAIN_CACHE)
        print(f"학습 블록 {len(ds)}개 준비 완료 -> 캐시 저장 ({TRAIN_CACHE})")

    return ds


def prepare_replay_dataset(tokenizer, path, n_needed):
    """영어 위키 복습 블록을 캐시에서 앞에서부터 n_needed개 읽거나, 없으면 만든다."""
    if os.path.isdir(path):
        ds = load_from_disk(path)
        if len(ds) < n_needed:
            raise SystemExit(
                f"{path} 에 영어 블록이 {len(ds)}개뿐인데 {n_needed}개가 필요합니다. "
                "다른 --replay_data 경로를 쓰거나 캐시를 지우고 다시 만드세요."
            )
        print(f"영어 복습 블록 {n_needed}개를 캐시에서 불러옴 ({path}, 전체 {len(ds)}개)")
        return ds.select(range(n_needed))

    print(f"영어 복습 블록 {n_needed}개 만드는 중 (wikimedia/wikipedia 20231101.en)")
    blocks = build_wiki_blocks(tokenizer, n_needed, config="20231101.en")
    ds = Dataset.from_dict({"input_ids": blocks})
    ds.save_to_disk(path)
    print(f"영어 복습 블록 {len(ds)}개 준비 완료 -> 캐시 저장 ({path})")
    return ds


def cast_state_dict(state_dict, dtype):
    """가중치의 CPU 복사본을 dtype으로 바꾼다. 원본 텐서(학습 중인 모델)는 그대로."""
    return {k: (v.detach().to("cpu", dtype=dtype) if v.is_floating_point()
                else v.detach().cpu())
            for k, v in state_dict.items()}


class JsonlLogCallback(TrainerCallback):
    """Trainer가 로그를 낼 때마다 한 줄씩 파일에 바로 쓴다 (학습이 중간에 끊겨도 남음).
    학습에는 영향 없음: 이미 계산된 loss, grad_norm, lr을 받아 적기만 한다."""

    def __init__(self, path):
        self.path = path
        open(path, "w", encoding="utf-8").close()   # 같은 이름으로 다시 돌리면 새로 씀

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **logs}, default=float) + "\n")


class CastSaveTrainer(Trainer):
    """체크포인트 파일만 save_dtype으로 저장하는 Trainer. 학습 과정은 Trainer와 똑같다."""

    def __init__(self, *a, save_dtype=torch.float16, **kw):
        super().__init__(*a, **kw)
        self.save_dtype = save_dtype

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        super()._save(output_dir, cast_state_dict(state_dict, self.save_dtype))
        # config.json의 dtype도 파일 쪽만 맞춰 준다 (메모리의 model.config는 그대로)
        cfg_path = os.path.join(output_dir or self.args.output_dir, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["dtype"] = str(self.save_dtype).replace("torch.", "")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def main():
    args = parse_args()
    output_dir = os.path.join("checkpoints", args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = prepare_train_dataset(tokenizer)
    n_ko, n_en = len(train_ds), 0
    if args.replay_data:
        # ratio = 영어 / (한국어 + 영어)  ->  영어 = 한국어 * ratio / (1 - ratio)
        n_en = round(n_ko * args.replay_ratio / (1 - args.replay_ratio))
        en_ds = prepare_replay_dataset(tokenizer, args.replay_data, n_en)
        # 한국어 뒤에 붙이기만 한다. 실제로 섞는 건 Trainer의 seed 셔플
        train_ds = concatenate_datasets([train_ds, en_ds.cast(train_ds.features)])

    # causal LM이라 labels는 input_ids와 동일
    train_ds = train_ds.map(lambda b: {"labels": b["input_ids"]}, batched=True)

    epoch_steps = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    max_steps = args.max_steps or epoch_steps
    print(f"[{args.run_name}] 한국어 {n_ko} + 영어 {n_en} = {len(train_ds)}블록 "
          f"(한 바퀴 {epoch_steps}스텝), warmup={args.warmup_steps}, "
          f"save_every={args.save_every}, seed={args.seed}, max_steps={max_steps}")

    # 나중에 이 런이 어떤 설정이었는지 체크포인트 옆에 남겨둔다
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "max_steps": max_steps, "n_ko_blocks": n_ko,
                   "n_en_blocks": n_en, "learning_rate": LEARNING_RATE,
                   "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM}, f, indent=2)

    # fp32로 불러와야 bf16 혼합정밀도 학습이 안정적으로 돌아감
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    targs = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,                    # 메모리 절약 + 속도
        gradient_checkpointing=True,  # 메모리 절약 (대신 약간 느려짐)
        optim="adamw_bnb_8bit",       # 옵티마이저 메모리 절약 (bitsandbytes 필요)
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_every,
        save_only_model=True,         # 옵티마이저 상태는 저장 안 함 (용량 절약)
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,     # Windows에서 안전
    )

    common = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,   # 체크포인트마다 토크나이저도 같이 저장
        callbacks=[JsonlLogCallback(os.path.join(output_dir, "train_log.jsonl"))],
    )
    if args.save_dtype == "float32":
        trainer = Trainer(**common)   # run01~03과 완전히 같은 저장 방식
    else:
        trainer = CastSaveTrainer(**common, save_dtype=getattr(torch, args.save_dtype))
    trainer.train()
    print(f"완료! 체크포인트 위치: {output_dir}")


if __name__ == "__main__":  # Windows에서 필수
    main()
