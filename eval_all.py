"""
forgetting-lab / eval_all.py

한 런(run)의 모든 체크포인트를 같은 조건으로 평가한다.
  * 영어: lm-eval (arc_easy, hellaswag, piqa, lambada_openai), fp16
  * 한국어: data/ko_heldout perplexity (eval_korean.py 로직 재사용), 같은 fp16
  * 영어 영역별: data/en_heldout_{wiki,books,web} perplexity, 같은 방식·fp16
    (한국어와 같이 재므로 모델은 체크포인트당 한 번만 올린다)
  * 원본 모델(step 0)도 같은 조건으로 한 번 평가해서 런끼리 공유한다
  * 이미 끝난 체크포인트는 건너뛴다 (중간에 끊겨도 이어서 돌리면 됨)

"이미 끝났다"의 기준 = 결과를 만든 체크포인트와 지금 체크포인트가 같은 파일일 때.
평가할 때마다 체크포인트 가중치 파일의 sha256 지문을 결과 옆에 같이 적어두고,
다음에 돌릴 때 지문이 다르면(같은 이름으로 다시 학습했으면) 다시 평가한다.
  * lm-eval: <run>/ckpt-<step>/ckpt_fingerprint.json
  * 한국어:  <run>/ko.csv 의 ckpt_fp 컬럼
  * 영어 영역별: <run>/en_domains.csv 의 ckpt_fp 컬럼
바뀐 체크포인트의 옛 결과는 지우지 않고 <run>/_stale/ 로 옮겨둔다.

지문이 없는 옛 결과(이 기능 전에 만든 것)는 기본적으로 믿지 않고 다시 평가한다.
결과가 맞다는 걸 이미 확인했다면 --trust_existing으로 지금 체크포인트의 지문을
붙여 그대로 쓴다. 단, 결과 파일이 체크포인트보다 먼저 만들어졌으면 그래도 다시 평가.

결과 위치 (fp32로 돌렸던 옛 결과는 results/fp32 에 따로 있음):
  results/fp16/base/step-0/...        <- 원본 모델, 런끼리 공유
  results/fp16/base/ko.csv
  results/fp16/<run>/ckpt-<step>/...  <- lm-eval 결과 JSON + 지문
  results/fp16/<run>/ko.csv           <- 한국어 perplexity + 지문
  results/fp16/<run>/en_domains.csv   <- 영어 영역별 perplexity + 지문

예)
  python eval_all.py --run run03_replay20
  python eval_all.py --run run01_ko_wiki --trust_existing   # 옛 결과에 지문만 붙이기
  python eval_all.py --run run01_ko_wiki --only en_domains  # 영어 영역별만 (나머지는 안 건드림)
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

from eval_korean import eval_loaded, free_cuda, load_blocks, load_model

BASE_MODEL = "EleutherAI/pythia-410m"
TASKS = ["arc_easy", "hellaswag", "piqa", "lambada_openai"]
FP_FILE = "ckpt_fingerprint.json"
EN_DOMAINS = {"wiki": "data/en_heldout_wiki", "books": "data/en_heldout_books",
              "web": "data/en_heldout_web"}
PARTS = ["lm_eval", "ko", "en_domains"]


def parse_args():
    p = argparse.ArgumentParser(description="한 런의 체크포인트 전체를 같은 조건으로 평가")
    p.add_argument("--run", required=True, help="checkpoints/<run> 의 런 이름")
    p.add_argument("--ckpt_root", default="checkpoints")
    p.add_argument("--results_root", default="results/fp16")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--dtype", default="float16", help="lm-eval과 한국어 평가 모두에 적용")
    p.add_argument("--steps", nargs="+", type=int, help="이 스텝들만 평가 (기본: 전부)")
    p.add_argument("--no_base", action="store_true", help="원본 모델 평가 건너뛰기")
    p.add_argument("--limit", type=int, help="lm-eval 샘플 수 제한 (테스트용)")
    p.add_argument("--ko_blocks", type=int, help="한국어 블록 수 제한 (테스트용)")
    p.add_argument("--en_blocks", type=int, help="영어 영역별 블록 수 제한 (테스트용)")
    p.add_argument("--only", nargs="+", choices=PARTS, default=PARTS,
                   help="이 평가만 돌린다 (기본: 전부). 나머지 결과 파일은 읽지도 쓰지도 않음")
    p.add_argument("--bs", default="8", help="lm-eval 배치 크기")
    p.add_argument("--ko_batch", type=int, default=4, help="한국어 평가 배치 크기")
    p.add_argument("--trust_existing", action="store_true",
                   help="지문 없는 옛 결과를 지금 체크포인트 것으로 인정하고 지문만 붙인다")
    return p.parse_args()


def find_checkpoints(ckpt_root, run, only=None):
    paths = glob.glob(os.path.join(ckpt_root, run, "checkpoint-*"))
    out = []
    for p in sorted(paths):
        m = re.search(r"checkpoint-(\d+)$", p.replace("\\", "/"))
        if m and (only is None or int(m.group(1)) in only):
            out.append((int(m.group(1)), p))
    return sorted(out)


# ---------- 체크포인트 지문 ----------

def weight_files(model_path):
    # 가중치 파일만. "*.bin"으로 찾으면 training_args.bin까지 잡혀서 안 됨
    files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    return files or sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin")))


def ckpt_fingerprint(model_path):
    """가중치 파일 내용의 sha256. 같은 이름으로 다시 학습하면 반드시 바뀐다.
    허브 모델(원본)은 로컬 파일이 아니므로 이름을 지문으로 쓴다."""
    if not os.path.isdir(model_path):
        return f"hub:{model_path}"
    files = weight_files(model_path)
    if not files:
        raise SystemExit(f"{model_path} 에 가중치 파일이 없습니다.")
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def ckpt_mtime(model_path):
    """체크포인트가 만들어진 시각 (허브 모델은 0 = 아주 옛날)."""
    if not os.path.isdir(model_path):
        return 0.0
    return max(os.path.getmtime(p) for p in weight_files(model_path))


def move_to_stale(path, run_dir, tag):
    """바뀐 체크포인트의 옛 결과를 지우지 않고 _stale/ 로 옮긴다."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(run_dir, "_stale", f"{tag}__{stamp}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(path, dest)
    print(f"    옛 결과 보관: {dest}")


# ---------- 영어: lm-eval ----------

def matching_results(out_dir, tasks, limit):
    """같은 조건(limit, 태스크)으로 돌린 결과 JSON들, 최신이 마지막."""
    hits = []
    for path in glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # 샘플 수를 제한해서 돌린 결과는 제한 없이 돌린 결과와 같지 않다
        if d.get("config", {}).get("limit") != limit:
            continue
        if all(t in d.get("results", {}) for t in tasks):
            hits.append(path)
    return sorted(hits, key=os.path.getmtime)


def write_stamp(out_dir, fp, model_path, result_json, limit, note):
    with open(os.path.join(out_dir, FP_FILE), "w", encoding="utf-8") as f:
        json.dump({"ckpt_fp": fp, "model": model_path, "limit": limit,
                   "result_json": os.path.relpath(result_json, out_dir).replace("\\", "/"),
                   "stamped_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "note": note}, f, indent=2, ensure_ascii=False)


def lm_eval_status(out_dir, fp, model_path, args):
    """반환: ("done", 이유) 또는 ("run", 이유). 필요하면 옛 결과를 _stale로 옮긴다."""
    if not os.path.isdir(out_dir):
        return "run", "결과 없음"
    hits = matching_results(out_dir, args.tasks, args.limit)
    stamp_path = os.path.join(out_dir, FP_FILE)

    if os.path.exists(stamp_path):
        with open(stamp_path, encoding="utf-8") as f:
            stamp = json.load(f)
        if stamp["ckpt_fp"] != fp:
            return "stale", f"체크포인트가 바뀜 ({stamp['ckpt_fp']} -> {fp})"
        rj = os.path.join(out_dir, stamp["result_json"])
        if stamp.get("limit") == args.limit and os.path.normpath(rj) in map(os.path.normpath, hits):
            return "done", "지문 일치"
        return "run", "같은 체크포인트지만 이 조건(limit/태스크)의 결과가 없음"

    # 지문이 없는 옛 결과
    if not hits:
        return "stale", "지문 없는 결과가 있지만 조건이 다름"
    if not args.trust_existing:
        return "stale", "지문 없는 옛 결과 (믿으려면 --trust_existing)"
    if os.path.getmtime(hits[-1]) < ckpt_mtime(model_path):
        return "stale", "옛 결과가 체크포인트보다 먼저 만들어짐 -> 믿을 수 없음"
    write_stamp(out_dir, fp, model_path, hits[-1], args.limit, "trust_existing로 인정")
    return "done", "옛 결과를 인정하고 지문을 붙임"


def run_lm_eval(model_path, out_dir, fp, args):
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype={args.dtype}",
        "--tasks", ",".join(args.tasks),
        "--device", "cuda:0",
        "--batch_size", str(args.bs),
        "--output_path", out_dir,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print("    $ " + " ".join(cmd[1:]))
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"lm-eval 실패 (exit {r.returncode}): {model_path}")
    new = [p for p in matching_results(out_dir, args.tasks, args.limit)
           if os.path.getmtime(p) >= t0 - 1]
    if not new:
        raise SystemExit(f"lm-eval은 끝났는데 결과 JSON을 못 찾았습니다: {out_dir}")
    write_stamp(out_dir, fp, model_path, new[-1], args.limit, "eval_all.py로 평가")


# ---------- 한국어: held-out perplexity ----------

KO_FIELDS = ["step", "model", "ko_loss", "ko_ppl", "dtype", "n_blocks", "ckpt_fp",
             "evaluated_at"]


def read_ko(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {int(row["step"]): row for row in csv.DictReader(f)}


def write_ko(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KO_FIELDS, restval="")
        w.writeheader()
        for step in sorted(rows):
            w.writerow({k: rows[step].get(k, "") for k in KO_FIELDS})


def stash_ko_row(run_dir, row, reason):
    """다시 재기 전의 한국어 결과를 _stale/ko_replaced.csv 에 남긴다."""
    path = os.path.join(run_dir, "_stale", "ko_replaced.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KO_FIELDS + ["replaced_reason"], restval="")
        if new_file:
            w.writeheader()
        w.writerow({**{k: row.get(k, "") for k in KO_FIELDS}, "replaced_reason": reason})


def ko_status(rows, step, fp, model_path, ko_csv, args, n_blocks):
    row = rows.get(step)
    if row is None:
        return "run", "결과 없음"
    if row["dtype"] != args.dtype or int(row["n_blocks"]) != n_blocks:
        return "stale", "정밀도나 블록 수가 다름"
    if row.get("ckpt_fp"):
        if row["ckpt_fp"] == fp:
            return "done", "지문 일치"
        return "stale", f"체크포인트가 바뀜 ({row['ckpt_fp']} -> {fp})"
    # 지문이 없는 옛 결과. 행마다 시각이 없어서 파일 시각으로만 거른다
    if not args.trust_existing:
        return "stale", "지문 없는 옛 결과 (믿으려면 --trust_existing)"
    if os.path.getmtime(ko_csv) < ckpt_mtime(model_path):
        return "stale", "ko.csv가 체크포인트보다 먼저 저장됨 -> 믿을 수 없음"
    row["ckpt_fp"] = fp
    return "adopt", "옛 결과를 인정하고 지문을 붙임"


# ---------- 영어 영역별: held-out perplexity ----------

EN_FIELDS = ["step", "model", "domain", "loss", "ppl", "ckpt_fp", "dtype", "n_blocks",
             "data_fp"]


def data_fingerprint(path):
    """held-out 세트 파일 내용의 sha256. 세트를 다시 만들면 바뀌어서 옛 결과를 다시 잰다."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(path, "data-*.arrow"))):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return "sha256:" + h.hexdigest()[:16]


def read_en(csv_path):
    """{(step, domain): row}"""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {(int(r["step"]), r["domain"]): r for r in csv.DictReader(f)}


def write_en(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EN_FIELDS, restval="")
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], list(EN_DOMAINS).index(k[1]))):
            w.writerow({k: rows[key].get(k, "") for k in EN_FIELDS})


def en_status(row, fp, args, n_blocks, dfp):
    if row is None:
        return "run", "결과 없음"
    if row["dtype"] != args.dtype or int(row["n_blocks"]) != n_blocks:
        return "stale", "정밀도나 블록 수가 다름"
    if row.get("data_fp") != dfp:
        return "stale", f"held-out 세트가 바뀜 ({row.get('data_fp') or '기록 없음'} -> {dfp})"
    if row["ckpt_fp"] != fp:
        return "stale", f"체크포인트가 바뀜 ({row['ckpt_fp']} -> {fp})"
    return "done", "지문 일치"


def stash_en_row(run_dir, row, reason):
    path = os.path.join(run_dir, "_stale", "en_domains_replaced.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EN_FIELDS + ["replaced_reason"], restval="")
        if new_file:
            w.writeheader()
        w.writerow({**{k: row.get(k, "") for k in EN_FIELDS}, "replaced_reason": reason})


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blocks = n_blocks = None
    if "ko" in args.only:
        blocks = load_blocks(limit=args.ko_blocks)
        n_blocks = len(blocks)
    en_sets, en_dfp = {}, {}
    if "en_domains" in args.only:
        for dom, path in EN_DOMAINS.items():
            if not os.path.isdir(path):
                raise SystemExit(f"{path} 가 없습니다. build_en_heldout.py를 먼저 돌리세요.")
            en_sets[dom] = load_blocks(path, limit=args.en_blocks)
            en_dfp[dom] = data_fingerprint(path)
    print(f"[{args.run}] 평가: {', '.join(args.only)} | dtype={args.dtype}, device={device}"
          + (f" | 한국어 {n_blocks}블록" if blocks is not None else "")
          + "".join(f" | en_{d} {len(b)}블록" for d, b in en_sets.items()))

    targets = []
    if not args.no_base and (args.steps is None or 0 in args.steps):
        targets.append((0, BASE_MODEL, os.path.join(args.results_root, "base"), "step-0"))
    pruned = []
    for step, path in find_checkpoints(args.ckpt_root, args.run, args.steps):
        if not weight_files(path):   # cleanup_checkpoints.py로 가중치를 지운 체크포인트
            pruned.append(step)
            continue
        targets.append((step, path, os.path.join(args.results_root, args.run),
                        f"ckpt-{step}"))
    if pruned:
        print(f"가중치를 정리한 체크포인트 {len(pruned)}개는 건너뜀 (기존 결과는 그대로): {pruned}")
    if not targets:
        raise SystemExit(f"{args.ckpt_root}/{args.run} 에서 체크포인트를 못 찾았습니다.")

    print(f"대상 {len(targets)}개: " + ", ".join(t[3] for t in targets) + "\n")

    for step, model_path, run_dir, tag in targets:
        print(f"=== step {step} ({model_path}) ===")
        out_dir = os.path.join(run_dir, tag)
        fp = ckpt_fingerprint(model_path)
        print(f"    지문: {fp}")

        # 영어 lm-eval (별도 프로세스)
        if "lm_eval" in args.only:
            status, why = lm_eval_status(out_dir, fp, model_path, args)
            if status == "done":
                print(f"    lm-eval: 건너뜀 ({why})")
            else:
                print(f"    lm-eval: {'다시 평가' if status == 'stale' else '평가'} ({why})")
                if status == "stale":
                    move_to_stale(out_dir, run_dir, tag)
                t0 = time.time()
                run_lm_eval(model_path, out_dir, fp, args)
                print(f"    lm-eval: {time.time() - t0:.0f}초")

        # 한국어·영어 영역별 perplexity: 필요한 게 있을 때만 모델을 한 번 올려서 같이 잰다
        model = None

        def get_model():
            nonlocal model
            if model is None:
                model = load_model(model_path, device, dtype)
            return model

        if "ko" in args.only:
            ko_csv = os.path.join(run_dir, "ko.csv")
            rows = read_ko(ko_csv)
            status, why = ko_status(rows, step, fp, model_path, ko_csv, args, n_blocks)
            if status in ("done", "adopt"):
                if status == "adopt":
                    write_ko(ko_csv, rows)
                print(f"    한국어: 건너뜀 ({why}, ppl={float(rows[step]['ko_ppl']):.3f})")
            else:
                print(f"    한국어: {'다시 평가' if status == 'stale' else '평가'} ({why})")
                if status == "stale":
                    stash_ko_row(run_dir, rows[step], why)
                t0 = time.time()
                loss, ppl = eval_loaded(get_model(), blocks, device, batch_size=args.ko_batch)
                rows = read_ko(ko_csv)  # 평가하는 동안 바뀌었을 수도 있으니 다시 읽음
                rows[step] = {"step": step, "model": model_path, "ko_loss": f"{loss:.6f}",
                              "ko_ppl": f"{ppl:.6f}", "dtype": args.dtype,
                              "n_blocks": n_blocks, "ckpt_fp": fp,
                              "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")}
                write_ko(ko_csv, rows)
                print(f"    한국어: loss={loss:.4f} ppl={ppl:.3f} ({time.time() - t0:.0f}초)")

        if "en_domains" in args.only:
            en_csv = os.path.join(run_dir, "en_domains.csv")
            for dom, dom_blocks in en_sets.items():
                rows = read_en(en_csv)
                row = rows.get((step, dom))
                status, why = en_status(row, fp, args, len(dom_blocks), en_dfp[dom])
                if status == "done":
                    print(f"    en_{dom}: 건너뜀 ({why}, ppl={float(row['ppl']):.3f})")
                    continue
                print(f"    en_{dom}: {'다시 평가' if status == 'stale' else '평가'} ({why})")
                if status == "stale":
                    stash_en_row(run_dir, row, why)
                t0 = time.time()
                loss, ppl = eval_loaded(get_model(), dom_blocks, device, batch_size=args.ko_batch)
                rows = read_en(en_csv)
                rows[(step, dom)] = {"step": step, "model": model_path, "domain": dom,
                                     "loss": f"{loss:.6f}", "ppl": f"{ppl:.6f}",
                                     "ckpt_fp": fp, "dtype": args.dtype,
                                     "n_blocks": len(dom_blocks), "data_fp": en_dfp[dom]}
                write_en(en_csv, rows)
                print(f"    en_{dom}: loss={loss:.4f} ppl={ppl:.3f} ({time.time() - t0:.0f}초)")

        if model is not None:
            model = None
            free_cuda()

    print(f"\n완료. 결과: {args.results_root}/{args.run} "
          f"(원본 모델은 {args.results_root}/base)")


if __name__ == "__main__":
    main()
