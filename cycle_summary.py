"""
forgetting-lab / cycle_summary.py

다회차 연구(LAB_NOTES 9장) 줄기 결과 요약. 줄기마다:

  1. retention 행렬 R[i, j] = cycle i 종료 시점의 언어 j held-out ppl (i = 0은 원본). 대각선 = 방금 배운 언어.
  2. 원래 지식: 영어 wiki/books/web ppl의 원본 대비 % 변화, probe KL(원본||체크포인트), 파라미터 거리.
  3. 비용: cycle마다 생성 GPU 초·생성 토큰, 학습 초·학습 토큰, 평가 초.

결과: 콘솔 + <results_root>/cycle_summary.md, <results_root>/retention.csv (모든 줄기의 긴 형식).

예)
  python cycle_summary.py                      # results/cycles 아래 줄기 전부
  python cycle_summary.py --trunks trunk_FT_s42 trunk_fresh_s42
"""
import argparse
import csv
import json
import os

LANGS = ["ko", "ja", "ru", "el"]
EN = ["en_wiki", "en_books", "en_web"]


def read_eval(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out.setdefault(r["tag"], {})[r["metric"]] = float(r["value"])
    return out


def cycle_tags(ev):
    return sorted((t for t in ev if t.startswith("c") and t != "c0_base"), key=lambda t: int(t[1:].split("_")[0]))


def main():
    p = argparse.ArgumentParser(description="다회차 연구 줄기 요약")
    p.add_argument("--results_root", default="results/cycles")
    p.add_argument("--trunks", nargs="+")
    args = p.parse_args()

    base = read_eval(os.path.join(args.results_root, "base", "eval.csv")).get("c0_base")
    if not base:
        raise SystemExit("원본(c0_base) 결과가 없습니다. trunk.py 또는 eval_cycle.py --ckpt base부터.")
    trunks = args.trunks or sorted(d for d in os.listdir(args.results_root)
                                   if d not in ("base", "_stale", "branches")   # 분기 결과는 c_check.py가 요약
                                   and os.path.isdir(os.path.join(args.results_root, d)))
    md, long_rows = [], []
    for t in trunks:
        ev = read_eval(os.path.join(args.results_root, t, "eval.csv"))
        cpath = os.path.join(args.results_root, t, "costs.json")
        costs = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else {}
        tags = ["c0_base"] + cycle_tags(ev)
        ev["c0_base"] = base
        md.append(f"## {t}\n")

        md.append("retention 행렬 R[i, j]: cycle i 종료 시점의 언어 j held-out ppl (굵게 = 그 cycle에 배운 언어)\n")
        md.append("| cycle \\ 언어 | " + " | ".join(LANGS) + " |")
        md.append("|---|" + "---|" * len(LANGS))
        for tag in tags:
            learned = tag.split("_", 1)[1] if tag != "c0_base" else None
            cells = []
            for lang in LANGS:
                v = ev[tag].get(f"ppl_{lang}")
                s = "-" if v is None else f"{v:.3f}"
                cells.append(f"**{s}**" if lang == learned else s)
                if v is not None:
                    long_rows.append({"trunk": t, "tag": tag, "metric": f"ppl_{lang}", "value": v})
            md.append(f"| {tag} | " + " | ".join(cells) + " |")

        md.append("\n원래 지식 (영어): 원본 대비 ppl % 변화, probe KL(원본||체크포인트) nats/토큰, 파라미터 거리\n")
        md.append("| cycle | " + " | ".join(EN) + " | KL(base‖ckpt) | KL(ckpt‖base) | ‖θ−θ0‖/‖θ0‖ |")
        md.append("|---|" + "---|" * (len(EN) + 3))
        for tag in tags:
            e = ev[tag]
            chg = [f"{(e[f'ppl_{d}'] / base[f'ppl_{d}'] - 1) * 100:+.1f}%" if f"ppl_{d}" in e else "-" for d in EN]
            extra = [f"{e[m]:.5f}" if m in e else "-" for m in ("kl_base_ckpt", "kl_ckpt_base", "param_dist")]
            md.append(f"| {tag} | " + " | ".join(chg + extra) + " |")
            for m in ("kl_base_ckpt", "kl_ckpt_base", "param_dist") + tuple(f"ppl_{d}" for d in EN):
                if m in e:
                    long_rows.append({"trunk": t, "tag": tag, "metric": m, "value": e[m]})

        if costs:
            md.append("\n비용\n")
            md.append("학습 연산량은 학습 토큰에 비례한다. FT는 복습이 없어 cycle당 120스텝, fresh는 150스텝 "
                      "(새 데이터 노출량은 같게, 9장 설계 결정).\n")
            md.append("| cycle | 생성 GPU 초 | 생성 토큰 | 학습 초 | 학습 토큰 (새 + 복습) | 평가 초 (마지막 실행) | GPU 최대 GiB |")
            md.append("|---|---|---|---|---|---|---|")
            tot = {"g": 0, "gt": 0, "tr": 0, "tt": 0, "nt": 0, "rt": 0}
            for tag, c in costs.items():
                md.append(f"| {tag} | {c.get('gen_gpu_seconds', 0):.0f} | {c.get('gen_tokens', 0):,} | "
                          f"{c.get('train_seconds', 0):.0f} | {c.get('train_tokens', 0):,} "
                          f"({c.get('new_tokens', 0):,} + {c.get('replay_tokens', 0):,}) | "
                          f"{c.get('eval_seconds_last_run', 0):.0f} | {c.get('peak_gpu_gib', '-')} |")
                for k, key in (("g", "gen_gpu_seconds"), ("gt", "gen_tokens"), ("tr", "train_seconds"),
                               ("tt", "train_tokens"), ("nt", "new_tokens"), ("rt", "replay_tokens")):
                    tot[k] += c.get(key, 0)
            md.append(f"| 합계 | {tot['g']:.0f} | {tot['gt']:,} | {tot['tr']:.0f} | {tot['tt']:,} "
                      f"({tot['nt']:,} + {tot['rt']:,}) | | |")
            cats = {tag: c["gen_categories"] for tag, c in costs.items() if "gen_categories" in c}
            if cats:
                md.append("\n생성 복습의 내용 구성 (전체 생성분, replay_quality 분류): "
                          + "; ".join(f"{tag} {v}" for tag, v in cats.items()))
        md.append("")

    text = "# 다회차 연구 줄기 요약\n\n" + "\n".join(md)
    print(text)
    with open(os.path.join(args.results_root, "cycle_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.results_root, "retention.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trunk", "tag", "metric", "value"])
        w.writeheader()
        w.writerows(long_rows)
    print(f"저장: {args.results_root}/cycle_summary.md, retention.csv")


if __name__ == "__main__":
    main()
