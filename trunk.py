"""
forgetting-lab / trunk.py

다회차 연구(LAB_NOTES 9장) B단계 "줄기" 실행: cycle 1(한국어) → 2(일본어) → 3(러시아어)를 이어서 학습·평가한다.

  * --mode FT    : 복습 없음. cycle마다 새 언어 3840블록만 (120스텝).
  * --mode fresh : cycle마다 시작 시점의 현재 체크포인트(cycle 1은 원본)에서 960블록을 새로 생성해서 복습
                   (새 3840 + 복습 960 = 150스텝). 생성분은 cycle끼리 재사용하지 않는다.
  * --mode base  : (C단계 C1, "base anchor 줄기") 3 cycle 내내 **원본 모델**에서 새 시드로 960블록을 생성해서 복습.
                   나머지(데이터, 150스텝, LR, 옵티마이저 초기화, 평가)는 fresh와 같다.
  * 생성 시드 = 줄기 시드 x 100000 + cycle x 1000 (+ 배치 번호), base 모드는 + 50000 (fresh와 겹치지 않게).
    학습 시드 = 줄기 시드.
    쓴 시드 구간은 <replay_root>/replay_seeds.json에 등록된다 (C단계 분기는 이 구간과 겹치는 시드를 못 씀).
  * 각 단계는 별도 프로세스(gen_replay.py, cycle_train.py, eval_cycle.py)로 돌리고, 하나라도 실패하면 멈춘다.
  * 원본(c0_base)은 <results_root>/base/eval.csv 에 한 번만 평가해서 줄기끼리 공유한다.
  * 이어서 돌리기: 끝난 단계는 건너뛴다.
      생성 = 복습 폴더에 source.json 있음 (중간이면 gen_replay가 배치 단위로 이어감)
      학습 = 체크포인트에 train_summary.json 있음 (중간이면 그 cycle 학습을 처음부터)
      평가 = eval_cycle.py가 지문으로 판단
  * 비용 기록: <results_root>/<줄기>/costs.json (cycle마다 생성 GPU 초·생성 토큰, 학습 초·학습 토큰, 평가 초).

위치: 체크포인트 <ckpt_root>/<줄기>/c<k>_<언어>, 복습 <replay_root>/<줄기>/c<k>_replay,
      결과 <results_root>/<줄기>/eval.csv

예)
  python trunk.py --mode FT --seed 42
  python trunk.py --mode fresh --seed 43
  python trunk.py --mode fresh --seed 1 --test --ckpt_root <임시> --replay_root <임시> --results_root <임시>
"""
import argparse
import json
import os
import subprocess
import sys
import time

# 9장 전용 언어 데이터 (build_lang_data.py). run01~04의 data/ko_train은 쓰지 않는다
LANG_DATA = {lang: f"data/cycles/{lang}_train" for lang in ("ko", "ja", "ru", "el")}
PY = sys.executable


# Windows 네이티브 강제 종료 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN). 로드 직후 첫 GPU 계산에서 가끔 난다 (9장 기록).
# 각 단계는 다시 돌려도 안전하다: 생성은 배치 단위로 이어가고, 학습은 끝에서만 저장하므로 처음부터, 평가는 지문으로 판단.
NATIVE_CRASH = {3221226505, -1073740791}
MAX_RETRY = 2
RETRY_LOG = os.environ.get("CYCLE_RETRY_LOG", "results/cycles/native_crash_retries.jsonl")   # 테스트는 환경변수로 바꿈


def run(cmd):
    print("  $ " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    for attempt in range(MAX_RETRY + 1):
        r = subprocess.run(cmd)
        if r.returncode in NATIVE_CRASH and attempt < MAX_RETRY:
            print(f"  (네이티브 강제 종료 exit {r.returncode}, 같은 단계를 다시 시도 {attempt + 1}/{MAX_RETRY})", flush=True)
            os.makedirs(os.path.dirname(RETRY_LOG), exist_ok=True)
            with open(RETRY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "exit": r.returncode,
                                    "attempt": attempt + 1, "cmd": cmd[1:]}, ensure_ascii=False) + "\n")
            continue
        break
    if r.returncode != 0:
        raise SystemExit(f"실패 (exit {r.returncode}): {' '.join(cmd[1:])}")
    return time.time() - t0


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="다회차 연구 줄기 실행 (cycle 1→2→3)")
    p.add_argument("--mode", choices=["FT", "fresh", "base"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--name", help="기본: trunk_<mode>_s<seed>")
    p.add_argument("--langs", nargs="+", default=["ko", "ja", "ru"], choices=list(LANG_DATA))
    p.add_argument("--new_n", type=int, default=3840)
    p.add_argument("--replay_n", type=int, default=960)
    p.add_argument("--gen_batch", type=int, default=48,
                   help="B단계는 48. C단계는 24 (48은 8GB에서 메모리가 넘쳐 느려짐, 9장)")
    p.add_argument("--ckpt_root", default="checkpoints/cycles")
    p.add_argument("--replay_root", default="data/cycles")
    p.add_argument("--results_root", default="results/cycles")
    p.add_argument("--test", action="store_true",
                   help="코드 점검용: 새 64 + 복습 16블록, 2스텝, 평가 8블록, 생성 배치 8")
    args = p.parse_args()
    name = args.name or f"trunk_{args.mode}_s{args.seed}"
    if args.test:
        args.new_n, args.replay_n, args.gen_batch = 64, 16, 8
    test_train = ["--max_steps", "2"] if args.test else []
    test_eval = ["--limit", "8"] if args.test else []

    res_dir = os.path.join(args.results_root, name)
    os.makedirs(res_dir, exist_ok=True)
    cfg_path = os.path.join(res_dir, "trunk_config.json")
    cfg = {"mode": args.mode, "seed": args.seed, "langs": args.langs, "new_n": args.new_n,
           "replay_n": args.replay_n if args.mode != "FT" else 0, "gen_batch": args.gen_batch,
           "test": args.test}
    if os.path.exists(cfg_path):
        old = load_json(cfg_path)
        # cycle을 뒤에 더 붙이는 것만 허용 (예: ko ja → ko ja ru)
        same = {k: v for k, v in old.items() if k != "langs"} == {k: v for k, v in cfg.items() if k != "langs"}
        if not same or args.langs[:len(old["langs"])] != old["langs"]:
            raise SystemExit(f"{name}: 예전 설정과 다릅니다 ({cfg_path}: {old}). 이름을 바꾸거나 확인하세요.")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    costs_path = os.path.join(res_dir, "costs.json")
    costs = load_json(costs_path) if os.path.exists(costs_path) else {}

    print(f"=== {name}: {args.mode}, seed {args.seed}, cycles {' → '.join(args.langs)}")
    run([PY, "eval_cycle.py", "--ckpt", "base", "--tag", "c0_base",
         "--results", os.path.join(args.results_root, "base", "eval.csv")] + test_eval)

    prev = "base"
    for k, lang in enumerate(args.langs, 1):
        tag = f"c{k}_{lang}"
        ck = os.path.join(args.ckpt_root, name, tag)
        c = costs.get(tag, {})
        print(f"\n--- cycle {k} ({lang}): {prev} → {ck}")
        replay = []
        if args.mode in ("fresh", "base"):
            rdir = os.path.join(args.replay_root, name, f"c{k}_replay")
            replay = [rdir]
            anchor = prev if args.mode == "fresh" and prev != "base" else "EleutherAI/pythia-410m"
            if os.path.exists(os.path.join(rdir, "source.json")):
                print(f"  생성: 건너뜀 (이미 있음 {rdir})")
            else:
                run([PY, "gen_replay.py", "--model", anchor,
                     "--out", rdir, "--n_blocks", str(args.replay_n), "--batch_size", str(args.gen_batch),
                     "--seed", str(args.seed * 100000 + (50000 if args.mode == "base" else 0) + k * 1000),
                     "--seed_registry", os.path.join(args.replay_root, "replay_seeds.json")])
            src = load_json(os.path.join(rdir, "source.json"))
            c.update({"gen_gpu_seconds": src["gen_gpu_seconds"], "gen_tokens": src["generated_tokens"],
                      "gen_categories": src["categories_all_generated"], "gen_from": src["model"]})
        if os.path.exists(os.path.join(ck, "train_summary.json")):
            print(f"  학습: 건너뜀 (이미 끝남 {ck})")
        else:
            run([PY, "cycle_train.py", "--init", prev, "--new_data", LANG_DATA[lang], "--new_n", str(args.new_n),
                 "--replay", *replay, "--replay_n", str(args.replay_n), "--seed", str(args.seed),
                 "--out", ck] + test_train)
        ts = load_json(os.path.join(ck, "train_summary.json"))
        c.update({"train_seconds": ts["train_seconds"], "train_tokens": ts["train_tokens"],
                  "new_tokens": ts["new_tokens"], "replay_tokens": ts["replay_tokens"],
                  "train_steps": ts["steps"], "peak_gpu_gib": ts["peak_gpu_gib"]})
        c["eval_seconds_last_run"] = round(run([PY, "eval_cycle.py", "--ckpt", ck, "--tag", tag,
                                                "--results", os.path.join(res_dir, "eval.csv")] + test_eval), 1)
        costs[tag] = c
        with open(costs_path, "w", encoding="utf-8") as f:
            json.dump(costs, f, indent=2, ensure_ascii=False)
        prev = ck
    print(f"\n=== {name} 완료. 결과 {res_dir}")


if __name__ == "__main__":
    main()
