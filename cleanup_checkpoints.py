"""
forgetting-lab / cleanup_checkpoints.py

남길 체크포인트(KEEP)를 빼고 나머지 체크포인트의 "가중치 파일"만 지운다.
작은 파일(config.json, trainer_state.json, training_args.bin, 토크나이저 등)과
run_config.json은 그대로 남아서 나중에도 어떤 설정이었는지 알 수 있다.

기본은 보고만 한다 (아무것도 안 지움):
  * 지울 목록과 확보되는 용량
  * 지울 체크포인트마다 평가 결과(lm-eval, 한국어, 영어 3영역)가 "지금 이 체크포인트의 것으로"
    저장돼 있는지 (지문 대조) 확인한 표
실제로 지우려면 --delete. 결과가 하나라도 빠진 체크포인트가 있으면 --delete여도 거부한다.

예)
  python cleanup_checkpoints.py              # 보고만
  python cleanup_checkpoints.py --delete     # 확인 후 실제 삭제
"""
import argparse
import csv
import glob
import json
import os

from eval_all import (EN_DOMAINS, ckpt_fingerprint, data_fingerprint, find_checkpoints,
                      weight_files)

DATA_FP = {dom: data_fingerprint(path) for dom, path in EN_DOMAINS.items()}

KEEP = {
    "run01_ko_wiki": {50, 100, 150, 200, 300},  # 충격, 불안정 전·중·후, 최종
    "run02_warmup100": {25, 125, 300},    # 망각 없는 학습, 충격, 최종
    "run03_replay20": {25, 75, 375},      # 충격, 충격, 최종
}
CKPT_ROOT = "checkpoints"
RESULTS_ROOT = "results/fp16"


def csv_fps(path, key):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {key(r): r.get("ckpt_fp", "") for r in csv.DictReader(f)}


def check(run, step, path):
    fp = ckpt_fingerprint(path)
    run_dir = os.path.join(RESULTS_ROOT, run)
    status = {}
    stamp = os.path.join(run_dir, f"ckpt-{step}", "ckpt_fingerprint.json")
    if os.path.exists(stamp):
        with open(stamp, encoding="utf-8") as f:
            s = json.load(f)
        ok = (s["ckpt_fp"] == fp and s.get("limit") is None
              and os.path.exists(os.path.join(run_dir, f"ckpt-{step}", s["result_json"])))
        status["lm-eval"] = ok
    else:
        status["lm-eval"] = False
    ko = csv_fps(os.path.join(run_dir, "ko.csv"), lambda r: int(r["step"]))
    status["한국어"] = ko.get(step) == fp
    # 영어 영역별은 체크포인트 지문 + 지금 held-out 세트 지문이 둘 다 맞아야 인정
    en_csv = os.path.join(run_dir, "en_domains.csv")
    en = {}
    if os.path.exists(en_csv):
        with open(en_csv, encoding="utf-8", newline="") as f:
            en = {(int(r["step"]), r["domain"]): r for r in csv.DictReader(f)}
    for dom, dpath in EN_DOMAINS.items():
        r = en.get((step, dom))
        status[f"en_{dom}"] = (r is not None and r["ckpt_fp"] == fp
                               and r.get("data_fp") == DATA_FP[dom])
    return status


def main():
    p = argparse.ArgumentParser(description="남길 것만 두고 체크포인트 가중치 정리")
    p.add_argument("--delete", action="store_true", help="실제로 지운다 (없으면 보고만)")
    args = p.parse_args()

    targets = []
    for run, keep in KEEP.items():
        for step, path in find_checkpoints(CKPT_ROOT, run):
            if step in keep:
                continue
            files = weight_files(path)
            if files:   # 이미 가중치를 지운 체크포인트는 건너뜀
                targets.append((run, step, path, files, sum(map(os.path.getsize, files))))

    if not targets:
        print("지울 체크포인트가 없습니다.")
        return

    cols = ["lm-eval", "한국어"] + [f"en_{d}" for d in EN_DOMAINS]
    print("지울 체크포인트 (가중치 파일만) - 평가 결과가 지금 체크포인트 것으로 저장돼 있는지:\n")
    print(f"{'런':<18}{'스텝':>5}{'용량(GB)':>10}  " + "  ".join(f"{c:>8}" for c in cols))
    missing = []
    for run, step, path, files, size in targets:
        st = check(run, step, path)
        print(f"{run:<18}{step:>5}{size / 2**30:>10.2f}  "
              + "  ".join(f"{'O' if st[c] else 'X':>8}" for c in cols))
        if not all(st.values()):
            missing.append((run, step, [c for c in cols if not st[c]]))

    total = sum(t[4] for t in targets)
    print(f"\n{len(targets)}개, 확보되는 용량 {total / 2**30:.1f} GB")
    print("남기는 것: " + ", ".join(f"{r} {sorted(k)}" for r, k in KEEP.items()))

    if missing:
        print("\n결과가 빠진 체크포인트가 있어 지우면 안 됩니다:")
        for run, step, cols_missing in missing:
            print(f"  {run} {step}: {', '.join(cols_missing)}")
        if args.delete:
            raise SystemExit("삭제 거부: 빠진 평가를 먼저 채우세요.")
        return

    if not args.delete:
        print("\n(보고만 했습니다. 실제로 지우려면 --delete)")
        return
    for run, step, path, files, size in targets:
        for f in files:
            os.remove(f)
        print(f"  삭제: {path} ({size / 2**30:.2f} GB, 작은 파일은 남김)")
    print(f"완료. {total / 2**30:.1f} GB 확보")


if __name__ == "__main__":
    main()
