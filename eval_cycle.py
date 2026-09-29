"""
forgetting-lab / eval_cycle.py

다회차 연구(LAB_NOTES 9장): 체크포인트 하나를 평가한다.

  * held-out ppl (fp16, eval_korean.eval_loaded와 같은 계산):
      영어 en_wiki / en_books / en_web, 언어별 ko / ja / ru / el (각 400블록)
  * 영어 probe KL: en_heldout_wiki에서 고정한 50블록 (0, 8, 16, …, 392번)에서 토큰마다
      kl_base_ckpt = KL(원본 || 이 체크포인트),  kl_ckpt_base = KL(이 체크포인트 || 원본)
    의 평균 (nats/토큰). 원본과 체크포인트를 둘 다 fp16으로 GPU에 올려 같은 입력으로 계산.
  * 파라미터 거리: ||θ − θ0|| / ||θ0|| (모든 가중치, fp32로 계산, θ0 = 허브의 원본 가중치).
    원본 자신은 0. 체크포인트는 fp16 파일이라 반올림 오차가 조금 섞인다.
  * lm-eval은 --lm_eval을 줄 때만 (기본 꺼짐). eval_all.py와 같은 태스크·설정.

지문 체계 (eval_all.py와 같음): 결과 csv의 행마다 체크포인트 가중치 sha256(ckpt_fp)과 데이터 지문(data_fp)을
적고, 다음에 돌릴 때 둘 다 같으면 건너뛴다. 다르면 옛 행을 _stale/eval_replaced.csv로 옮기고 다시 잰다.

예)
  python eval_cycle.py --ckpt base --tag c0_base --results results/cycles/base/eval.csv
  python eval_cycle.py --ckpt checkpoints/cycles/trunk_FT_s42/c1_ko --tag c1_ko --results results/cycles/trunk_FT_s42/eval.csv
  python eval_cycle.py --ckpt base --tag t --results <임시.csv> --limit 8      # 테스트
"""
import argparse
import csv
import datetime
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from eval_all import TASKS, ckpt_fingerprint, data_fingerprint, lm_eval_status, move_to_stale, run_lm_eval
from eval_korean import eval_loaded, free_cuda, load_blocks

BASE_MODEL = "EleutherAI/pythia-410m"
SETS = {"en_wiki": "data/en_heldout_wiki", "en_books": "data/en_heldout_books", "en_web": "data/en_heldout_web",
        # 9장 전용 언어 데이터 (build_lang_data.py). run01~04의 data/ko_heldout과는 다른 세트다
        "ko": "data/cycles/ko_heldout", "ja": "data/cycles/ja_heldout", "ru": "data/cycles/ru_heldout",
        "el": "data/cycles/el_heldout"}
PROBE_SET, PROBE_IDX = "data/en_heldout_wiki", list(range(0, 400, 8))   # 50블록 고정
FIELDS = ["tag", "model", "metric", "value", "n_blocks", "dtype", "ckpt_fp", "data_fp", "evaluated_at"]


def read_rows(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["tag"], r["metric"]): r for r in csv.DictReader(f)}


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({f: rows[k].get(f, "") for f in FIELDS})


def stash(path, row, reason):
    stale = os.path.join(os.path.dirname(path) or ".", "_stale", "eval_replaced.csv")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    new = not os.path.exists(stale)
    with open(stale, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["replaced_reason"])
        if new:
            w.writeheader()
        w.writerow({**{k: row.get(k, "") for k in FIELDS}, "replaced_reason": reason})


@torch.no_grad()
def probe_kl(model, base, blocks, batch=4):
    """토큰 평균 KL(base||ckpt), KL(ckpt||base). nats/토큰."""
    s_bc = s_cb = n = 0.0
    for i in range(0, len(blocks), batch):
        ids = blocks[i:i + batch].to("cuda")
        lc = F.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
        lb = F.log_softmax(base(input_ids=ids).logits[:, :-1].float(), -1)
        s_bc += (lb.exp() * (lb - lc)).sum().item()
        s_cb += (lc.exp() * (lc - lb)).sum().item()
        n += lc.shape[0] * lc.shape[1]
        del lc, lb
    return s_bc / n, s_cb / n


@torch.no_grad()
def param_distance(ckpt_path):
    theta0 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).state_dict()
    theta = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.float32).state_dict()
    num = den = 0.0
    for k, v0 in theta0.items():
        if not v0.is_floating_point():
            continue
        num += (theta[k] - v0).double().pow(2).sum().item()
        den += v0.double().pow(2).sum().item()
    return (num / den) ** 0.5


def main():
    p = argparse.ArgumentParser(description="다회차 연구: 체크포인트 하나 평가")
    p.add_argument("--ckpt", required=True, help="체크포인트 폴더, 또는 base")
    p.add_argument("--tag", required=True, help="결과 행 이름 (예: c1_ko)")
    p.add_argument("--results", required=True, help="결과 csv")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--no_probe", action="store_true", help="probe KL·파라미터 거리 건너뛰기")
    p.add_argument("--lm_eval", action="store_true", help="lm-eval도 (기본 꺼짐)")
    p.add_argument("--lm_eval_limit", type=int, help="lm-eval 샘플 수 제한 (테스트용)")
    p.add_argument("--limit", type=int, help="각 세트 앞 N블록만 (테스트용). probe도 앞 N블록")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    path = BASE_MODEL if args.ckpt == "base" else args.ckpt
    fp = ckpt_fingerprint(path)
    dtype = getattr(torch, args.dtype)
    print(f"[eval_cycle] {args.tag}: {path} ({fp})")

    todo = {}   # metric -> (data 경로 또는 None, n_blocks, data_fp)
    for s in args.sets:
        n = len(load_blocks(SETS[s], limit=args.limit))
        todo[f"ppl_{s}"] = (SETS[s], n, data_fingerprint(SETS[s]))
    if not args.no_probe:
        n = len(PROBE_IDX[:args.limit] if args.limit else PROBE_IDX)
        dfp = data_fingerprint(PROBE_SET) + f"#idx0:400:8"
        todo["kl_base_ckpt"] = todo["kl_ckpt_base"] = (PROBE_SET, n, dfp)
        todo["param_dist"] = (None, 0, "")

    rows = read_rows(args.results)
    run = []
    for m, (_, n, dfp) in todo.items():
        r = rows.get((args.tag, m))
        if r is None:
            run.append(m)
        elif (r["ckpt_fp"], r["data_fp"], int(r["n_blocks"]), r["dtype"]) == (fp, dfp, n, args.dtype):
            print(f"  {m}: 건너뜀 (지문 일치, {float(r['value']):.4f})")
        else:
            reason = "체크포인트·데이터·블록 수·정밀도 중 바뀐 것이 있음"
            print(f"  {m}: 다시 잼 ({reason})")
            stash(args.results, r, reason)
            run.append(m)

    def put(metric, value):
        rows = read_rows(args.results)
        _, n, dfp = todo[metric]
        rows[(args.tag, metric)] = {"tag": args.tag, "model": path, "metric": metric, "value": f"{value:.6f}",
                                    "n_blocks": n, "dtype": args.dtype, "ckpt_fp": fp, "data_fp": dfp,
                                    "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")}
        write_rows(args.results, rows)

    t_start = time.time()
    ppl_todo = [m for m in run if m.startswith("ppl_")]
    kl_todo = [m for m in run if m.startswith("kl_")]
    if ppl_todo or kl_todo:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to("cuda").eval()
        for m in ppl_todo:
            t0 = time.time()
            loss, ppl = eval_loaded(model, load_blocks(todo[m][0], limit=args.limit), "cuda", batch_size=args.batch)
            put(m, ppl)
            print(f"  {m}: {ppl:.4f} ({time.time() - t0:.0f}초)")
        if kl_todo:
            t0 = time.time()
            if args.ckpt == "base":
                kl_bc = kl_cb = 0.0
            else:
                base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to("cuda").eval()
                probe = load_blocks(PROBE_SET)[PROBE_IDX][:args.limit or None]
                kl_bc, kl_cb = probe_kl(model, base, probe, args.batch)
                base = None
            put("kl_base_ckpt", kl_bc)
            put("kl_ckpt_base", kl_cb)
            print(f"  probe KL(base||ckpt) {kl_bc:.5f}, KL(ckpt||base) {kl_cb:.5f} nats/토큰 ({time.time() - t0:.0f}초)")
        model = None
        free_cuda()
    if "param_dist" in run:
        d = 0.0 if args.ckpt == "base" else param_distance(path)
        put("param_dist", d)
        print(f"  param_dist ||θ−θ0||/||θ0|| = {d:.6f}")

    if args.lm_eval:
        out_dir = os.path.join(os.path.dirname(args.results) or ".", args.tag, "lm_eval")
        la = SimpleNamespace(tasks=TASKS, limit=args.lm_eval_limit, trust_existing=False, dtype=args.dtype, bs="8")
        status, why = lm_eval_status(out_dir, fp, path, la)
        if status == "done":
            print(f"  lm-eval: 건너뜀 ({why})")
        else:
            if status == "stale":
                move_to_stale(out_dir, os.path.dirname(out_dir), "lm_eval")
            run_lm_eval(path, out_dir, fp, la)
    print(f"  평가 {time.time() - t_start:.0f}초")


if __name__ == "__main__":
    main()
