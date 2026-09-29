"""
forgetting-lab / branch.py

다회차 연구(LAB_NOTES 9장) C단계 분기 실행: fresh 줄기의 M_3(c3_ru)에서 그리스어 cycle 하나를 arm별로 학습·평가한다.

공통 (모든 arm): 시작점 = <줄기>/c3_ru, 새 데이터 = data/cycles/el_train 3840블록, 150스텝, LR 1.5e-5 (워밍업 10 후 고정),
옵티마이저 초기화, **학습 시드 = 줄기 시드** (전체 4800블록이라 셔플 순서가 arm끼리 같다 → 그리스어 순서와 복습 슬롯
위치가 같고 복습 내용만 다르다). 복습은 정확히 960블록 (512토큰 고정이라 토큰 수도 같음).

arm (C2 = 2x2 + 통제, C3 = 나이·혼합):
  latest_full : 최신(M_3)에서 960블록 생성, 전체
  latest_ep   : 최신(M_3)에서 영어 산문만 960블록 (gen_replay --english_only: 필터 통과 960개가 찰 때까지 추가 생성)
  base_full   : 원본에서 960블록, 전체
  base_ep     : 원본에서 영어 산문만 960블록
  ft150       : 생성 없음. 복습 자리를 그리스어 학습 3840블록 중 무작위 960블록 반복으로 채움 (cycle_train --new_repeat)
  m1_full     : (C3) 그 줄기의 M_1(c1_ko)에서 960블록, 전체
  m2_full     : (C3) 그 줄기의 M_2(c2_ja)에서 960블록, 전체
  mix         : (C3) 원본 전체 480 + 최신 전체 480 (각각 새 시드로 따로 생성, cycle_train --replay 두 폴더 50/50)
  latest_kl   : (C4) latest_full이 쓴 복습 960블록을 그대로 재사용, 목적 함수만 KL (teacher = 그 복습을 생성한 M_3)
  base_kl     : (C4) base_full이 쓴 복습 960블록을 그대로 재사용, 목적 함수만 KL (teacher = 원본)
    C4 arm은 생성하지 않는다. 재사용은 분기 replay 규칙의 명시적 예외로 시드 등록부의 원래 항목에 "reused_by"로 적는다
    (규칙은 "학생이 이미 학습한 복습"을 막는 것. 이 글은 형제 분기가 학습했을 뿐 M_3는 본 적 없음).
    teacher 지문·토크나이저·구조 확인은 cycle_train.py가 한다 (다르면 멈춤).

생성 시드 = 줄기 시드 x 100000 + 70000 + arm 번호 x 1000 (줄기·base 줄기의 생성 구간과 겹치지 않음, 등록부로 강제).
생성 배치 기본 24 (48은 8GB에서 KV 캐시가 넘쳐 매우 느려짐, 9장). 생성 코드(gen_replay.py 등)는 B단계와 같은 파일.

위치: 체크포인트 <ckpt_root>/branches/<줄기>/<arm>, 복습 <replay_root>/branches/<줄기>/<arm>_replay,
      결과 <results_root>/branches/<줄기>/eval.csv (tag = arm), 비용 costs.json.
이어서 돌리기는 trunk.py와 같은 규칙 (생성 = source.json, 학습 = train_summary.json, 평가 = 지문).

예)
  python branch.py --trunk trunk_fresh_s42                       # C2 arm 다섯 개
  python branch.py --trunk trunk_fresh_s42 --arms m1_full m2_full mix   # C3
  python branch.py --trunk trunk_fresh_s1 --test --ckpt_root <임시> --replay_root <임시> --results_root <임시>
"""
import argparse
import json
import os

from trunk import LANG_DATA, PY, load_json, run

BASE = "EleutherAI/pythia-410m"
# arm: (복습 출처 목록 [(anchor, english_only, 블록 수, 시드 번호)], new_repeat)
ARMS = {
    "latest_full": ([("M3", False, 960, 1)], 0),
    "latest_ep": ([("M3", True, 960, 2)], 0),
    "base_full": ([("base", False, 960, 3)], 0),
    "base_ep": ([("base", True, 960, 4)], 0),
    "ft150": ([], 960),
    "m1_full": ([("M1", False, 960, 5)], 0),
    "m2_full": ([("M2", False, 960, 6)], 0),
    "mix": ([("base", False, 480, 7), ("M3", False, 480, 8)], 0),
}
REUSE_KL = {"latest_kl": "latest_full", "base_kl": "base_full"}   # C4: 복습 재사용 + KL 목적 함수
C2_ARMS = ["latest_full", "latest_ep", "base_full", "base_ep", "ft150"]
C3_ARMS = ["m1_full", "m2_full", "mix"]
C4_ARMS = list(REUSE_KL)
REUSE_REASON = ("C4 matched-data objective comparison: 형제 분기(NTP arm)가 쓴 복습을 KL로 다시 씀. 재사용 금지 규칙은 "
                "학생이 이미 학습한 복습을 막는 것이고, 이 복습은 분기 출발점 M_3가 학습한 적 없음 (명시적 예외, 2026-09-27)")


def allow_reuse(registry, rdir, arm, trunk):
    """시드 등록부에서 rdir을 만든 항목에 재사용 예외를 적는다 (생성은 하지 않으므로 시드 구간은 그대로)."""
    with open(registry, encoding="utf-8") as f:
        entries = json.load(f)
    norm = lambda p: os.path.normcase(os.path.abspath(p))
    hit = [e for e in entries if norm(e["out"]) == norm(rdir)]
    if len(hit) != 1:
        raise SystemExit(f"시드 등록부({registry})에서 {rdir} 항목을 찾지 못함 ({len(hit)}개). 멈춤.")
    uses = hit[0].setdefault("reused_by", [])
    if not any(u["arm"] == arm and u["trunk"] == trunk for u in uses):
        uses.append({"trunk": trunk, "arm": arm, "objective": "kl", "reason": REUSE_REASON})
        with open(registry, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
ANCHOR_TAG = {"M1": "c1_ko", "M2": "c2_ja", "M3": "c3_ru"}
NEW_LANG = "el"


def main():
    p = argparse.ArgumentParser(description="다회차 연구 C단계 분기 (M_3에서 그리스어)")
    p.add_argument("--trunk", required=True, help="fresh 줄기 이름 (예: trunk_fresh_s42)")
    p.add_argument("--arms", nargs="+", default=C2_ARMS, choices=list(ARMS) + C4_ARMS)
    p.add_argument("--gen_batch", type=int, default=24)
    p.add_argument("--ckpt_root", default="checkpoints/cycles")
    p.add_argument("--replay_root", default="data/cycles")
    p.add_argument("--results_root", default="results/cycles")
    p.add_argument("--test", action="store_true", help="코드 점검용: 새 64 + 복습 16블록, 2스텝, 평가 8블록, 생성 배치 8")
    args = p.parse_args()

    tcfg = load_json(os.path.join(args.results_root, args.trunk, "trunk_config.json"))
    if tcfg["mode"] != "fresh":
        raise SystemExit(f"{args.trunk}은 fresh 줄기가 아닙니다 ({tcfg['mode']}).")
    seed = tcfg["seed"]
    new_n, replay_n, gen_batch = (64, 16, 8) if args.test else (3840, 960, args.gen_batch)
    scale = replay_n / 960
    anchors = {"base": BASE, **{k: os.path.join(args.ckpt_root, args.trunk, t) for k, t in ANCHOR_TAG.items()}}
    m3 = anchors["M3"]
    if not os.path.exists(os.path.join(m3, "train_summary.json")):
        raise SystemExit(f"{m3} 가 없습니다. 줄기 {args.trunk}이 끝났는지 확인하세요.")

    res_dir = os.path.join(args.results_root, "branches", args.trunk)
    os.makedirs(res_dir, exist_ok=True)
    cfg_path = os.path.join(res_dir, "branch_config.json")
    cfg = {"trunk": args.trunk, "seed": seed, "start": m3, "new_lang": NEW_LANG, "new_n": new_n,
           "replay_n": replay_n, "gen_batch": gen_batch, "test": args.test}
    if os.path.exists(cfg_path) and load_json(cfg_path) != cfg:
        raise SystemExit(f"분기 설정이 예전과 다릅니다 ({cfg_path}: {load_json(cfg_path)}).")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    costs_path = os.path.join(res_dir, "costs.json")
    costs = load_json(costs_path) if os.path.exists(costs_path) else {}
    test_train = ["--max_steps", "2"] if args.test else []
    test_eval = ["--limit", "8"] if args.test else []

    print(f"=== 분기 {args.trunk} (seed {seed}): {m3} → 그리스어, arm {args.arms}")
    run([PY, "eval_cycle.py", "--ckpt", "base", "--tag", "c0_base",
         "--results", os.path.join(args.results_root, "base", "eval.csv")] + test_eval)
    for arm in args.arms:
        loss = "kl" if arm in REUSE_KL else "ntp"
        sources, repeat = ARMS.get(arm, ([], 0))
        ck = os.path.join(args.ckpt_root, "branches", args.trunk, arm)
        c = {"arm": arm, "objective": loss}
        print(f"\n--- arm {arm}: {m3} → {ck}")
        replay = []
        if arm in REUSE_KL:
            src_arm = REUSE_KL[arm]
            rdir = os.path.join(args.replay_root, "branches", args.trunk, f"{src_arm}_replay")
            if not os.path.exists(os.path.join(rdir, "source.json")):
                raise SystemExit(f"{rdir} 가 없습니다. C2의 {src_arm} arm이 먼저 끝나야 합니다.")
            allow_reuse(os.path.join(args.replay_root, "replay_seeds.json"), rdir, arm, args.trunk)
            replay = [rdir]
            src = load_json(os.path.join(rdir, "source.json"))
            c["reused_replay"] = {"from_arm": src_arm, "dir": rdir, "teacher": src["model"],
                                  "teacher_fp": src.get("ckpt_fp"), "seed": src["seed"]}
            print(f"  복습 재사용 (생성 없음): {rdir}, teacher {src['model']}")
        for anchor, ep, n, idx in sources:
            suffix = "" if len(sources) == 1 else f"_{anchor.lower()}"
            rdir = os.path.join(args.replay_root, "branches", args.trunk, f"{arm}{suffix}_replay")
            replay.append(rdir)
            if os.path.exists(os.path.join(rdir, "source.json")):
                print(f"  생성: 건너뜀 (이미 있음 {rdir})")
            else:
                run([PY, "gen_replay.py", "--model", anchors[anchor], "--out", rdir,
                     "--n_blocks", str(int(n * scale)), "--batch_size", str(gen_batch),
                     "--seed", str(seed * 100000 + 70000 + idx * 1000),
                     "--seed_registry", os.path.join(args.replay_root, "replay_seeds.json")]
                    + (["--english_only"] if ep else []))
            src = load_json(os.path.join(rdir, "source.json"))
            c.setdefault("gen", []).append({"from": src["model"], "english_only": src["english_only"],
                                            "kept_blocks": src["n_blocks"], "generated_blocks": src["generated_blocks"],
                                            "generated_tokens": src["generated_tokens"],
                                            "gen_gpu_seconds": src["gen_gpu_seconds"],
                                            "categories_all_generated": src["categories_all_generated"]})
        if os.path.exists(os.path.join(ck, "train_summary.json")):
            print(f"  학습: 건너뜀 (이미 끝남 {ck})")
        else:
            run([PY, "cycle_train.py", "--init", m3, "--new_data", LANG_DATA[NEW_LANG], "--new_n", str(new_n),
                 "--replay", *replay, "--replay_n", str(replay_n), "--replay_loss", loss,
                 "--new_repeat", str(int(repeat * scale)), "--seed", str(seed), "--out", ck] + test_train)
        ts = load_json(os.path.join(ck, "train_summary.json"))
        c.update({k: ts[k] for k in ("train_seconds", "train_tokens", "new_tokens", "replay_tokens",
                                     "repeat_tokens", "steps", "peak_gpu_gib") if k in ts})
        c["eval_seconds_last_run"] = round(run([PY, "eval_cycle.py", "--ckpt", ck, "--tag", arm,
                                                "--results", os.path.join(res_dir, "eval.csv")] + test_eval), 1)
        costs[arm] = c
        with open(costs_path, "w", encoding="utf-8") as f:
            json.dump(costs, f, indent=2, ensure_ascii=False)
    print(f"\n=== 분기 {args.trunk} 완료. 결과 {res_dir}")


if __name__ == "__main__":
    main()
