"""
forgetting-lab / verify_draft.py

PAPER_DRAFT.md의 수치를 결과 파일에서 다시 읽거나 다시 계산해서 자동으로 대조한다. 아무것도 학습·평가하지 않는다.

  * 원천: results/cycles/*/eval.csv (ppl, probe KL, 파라미터 거리), c_check.json, gate_check.json, 체크포인트의
    train_log.jsonl·run_config.json·train_summary.json, 복습 폴더의 source.json·블록, data/cycles/build_report.json,
    코드 상수 (build_lang_data.py, eval_cycle.py, gen_selfgen.py).
  * c_check.json에 없는 수치(기울기 평균, 사후 잔차, 복습 구성, 중복 블록, 13-gram 겹침, 글 수 등)는 원본에서 다시 계산한다.
  * 비교: 본문에 적힌 자릿수로 반올림했을 때 같으면 일치 (|본문 − 계산| ≤ 0.5 x 10^-자릿수). "about"이 붙은 값은 따로 허용 범위를 둔다.
  * 판정 문구 (Supported, Partial, Mixed, Category A 등)도 c_check.json의 판정과 대조한다.
  * 마지막에, 어떤 검사에도 걸리지 않은 본문 숫자를 "검증 안 된 수치"로 문맥과 함께 보고한다 (절 번호·연도·그림 번호 등은 제외).

  * 외부 논문에서 인용한 수치(원문 확인값)는 "외부 인용" 검사로 본문 값이 바뀌지 않았는지만 본다.
  * --release: 공개 직전 검사. 내부 메모("All summaries: recheck ...", "[GitHub URL to be added]", 서지 확인 태그 등)와
    references.bib의 미완성 저자 목록("and others")이 남아 있으면 실패. 평소에는 목록만 보고한다.

결과: 콘솔 + results/verify_draft_report.txt. 불일치나 누락이 있으면 exit 1.

예)
  python verify_draft.py
  python verify_draft.py --source paper/main.tex --release
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter

from datasets import load_from_disk
from transformers import AutoTokenizer

from cycle_summary import read_eval
from replay_quality import gram_set, ngrams, rep4

DRAFT = "PAPER_DRAFT.md"
R = "results/cycles"
EN = ["en_wiki", "en_books", "en_web"]
MID = ["ko", "ja", "ru"]
S = [42, 43]
# 생성 동등성 검증 결과 (C단계 준비 때 스크래치에서 잰 것을 results로 옮김)
SCRATCH_EQ = [p for p in ["results/cycles/generation_equivalence.json"] if os.path.exists(p)]

# ---------------------------------------------------------------- 원천 읽기
base = read_eval(f"{R}/base/eval.csv")["c0_base"]
T = {f"{m}{s}": read_eval(f"{R}/trunk_{m}_s{s}/eval.csv") for m in ("FT", "fresh", "base") for s in S}
B = {s: read_eval(f"{R}/branches/trunk_fresh_s{s}/eval.csv") for s in S}
CC = json.load(open(f"{R}/c_check.json", encoding="utf-8"))
GATE = json.load(open(f"{R}/gate_check.json", encoding="utf-8"))
BUILD = {r["lang"]: r for r in json.load(open("data/cycles/build_report.json", encoding="utf-8"))}
KLF = json.load(open(f"{R}/kl_floor.json", encoding="utf-8"))


def pct(a, b):
    return (a / b - 1) * 100


def en(ev):
    return {d: pct(ev[f"ppl_{d}"], base[f"ppl_{d}"]) for d in EN}


def enm(ev):
    return st.mean(en(ev).values())


def midm(ev, ref):
    return st.mean(pct(ev[f"ppl_{l}"], ref[f"ppl_{l}"]) for l in MID)


M3 = {s: T[f"fresh{s}"]["c3_ru"] for s in S}
mean2 = lambda f: st.mean(f(s) for s in S)


def log_mean(s, arm, key):
    rows = [json.loads(l) for l in open(f"checkpoints/cycles/branches/trunk_fresh_s{s}/{arm}/train_log.jsonl",
                                        encoding="utf-8") if f'"{key}"' in l]
    return st.mean(r[key] for r in rows), rows


def src(path):
    return json.load(open(os.path.join(path, "source.json"), encoding="utf-8"))


# ---------------------------------------------------------------- c_check에 없는 값 다시 계산
recompute = {}
# 사후 잔차: 실제 − (base_full + 교사 체크포인트의 영어 저하)
recompute["residuals"] = {s: [enm(B[s][a]) - enm(B[s]["base_full"]) - enm(T[f"fresh{s}"][t])
                              for a, t in (("m1_full", "c1_ko"), ("m2_full", "c2_ja"), ("latest_full", "c3_ru"))] for s in S}
# 복습 구성: B단계 fresh 줄기 복습의 비영어 블록 수
nonen = {k: [src(f"data/cycles/trunk_fresh_s{s}/c{k}_replay")["categories_all_generated"]["비영어"] for s in S] for k in (1, 3)}
nblocks = {src(f"data/cycles/trunk_fresh_s{s}/c{k}_replay")["n_blocks"] for s in S for k in (1, 2, 3)}
# KL 이득 비
gain = {s: (enm(B[s]["latest_full"]) - enm(B[s]["latest_kl"]), enm(B[s]["base_full"]) - enm(B[s]["base_kl"])) for s in S}
gain_dom = {d: st.mean(en(B[s]["latest_full"])[d] - en(B[s]["latest_kl"])[d] for s in S) /
               st.mean(en(B[s]["base_full"])[d] - en(B[s]["base_kl"])[d] for s in S) for d in EN}
# 학습 로그
gn = {(s, a): log_mean(s, a, "grad_norm")[0] for s in S for a in ("latest_full", "latest_kl")}
rkl = {a: st.mean(log_mean(s, a, "replay_kl")[0] for s in S) for a in ("latest_kl", "base_kl")}
kl_floor_latest = [log_mean(s, "latest_kl", "replay_kl")[1][0]["replay_kl"] for s in S]   # 1스텝: 학생 = teacher
# 원본 생성 복습의 코드·망가진 블록 비율 (이 연구의 원본 생성 복습 전부, 필터 없는 것)
base_replays = ([f"data/cycles/trunk_fresh_s{s}/c1_replay" for s in S] + [f"data/cycles/trunk_base_s{s}/c{k}_replay" for s in S for k in (1, 2, 3)]
                + [f"data/cycles/branches/trunk_fresh_s{s}/base_full_replay" for s in S])
code_n = sum(src(p)["categories_all_generated"]["코드"] for p in base_replays)
gen_n = sum(src(p)["generated_blocks"] for p in base_replays)
broken = [rep4(list(b)) >= 0.5 for p in base_replays for b in load_from_disk(p)["input_ids"]]
# 공백만 있는 똑같은 블록
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
dup = {}
for s in S:
    for arm in ("latest_full", "base_full"):
        rows = [tuple(r) for r in load_from_disk(f"data/cycles/branches/trunk_fresh_s{s}/{arm}_replay")["input_ids"]]
        dup[(s, arm)] = [(n, tok.decode(list(k)).strip() == "") for k, n in Counter(rows).items() if n > 1]
# 옛 한국어 held-out 13-gram 겹침
old_tr = [list(b) for b in load_from_disk("data/ko_train")["input_ids"]]
old_he = [list(b) for b in load_from_disk("data/ko_heldout")["input_ids"]]
g = gram_set(old_tr)
hits = sum(sum(x in g for x in ngrams(h)) for h in old_he)
old_ko_overlap = hits / sum(len(ngrams(h)) for h in old_he) * 100
# 생성 배치
gen_batch = {"B": {src(f"data/cycles/trunk_fresh_s{s}/c{k}_replay")["batch_size"] for s in S for k in (1, 2, 3)},
             "C": {src(p)["batch_size"] for p in glob.glob("data/cycles/branches/*/*_replay") + glob.glob("data/cycles/trunk_base_s*/c*_replay")}}
# 학습 설정 (실제 run_config / train_summary)
cfgs = [json.load(open(p, encoding="utf-8")) for p in glob.glob("checkpoints/cycles/**/run_config.json", recursive=True)]
sums = {os.path.dirname(p): json.load(open(p, encoding="utf-8")) for p in glob.glob("checkpoints/cycles/**/train_summary.json", recursive=True)}
steps_ft = {v["steps"] for k, v in sums.items() if "trunk_FT" in k}
steps_rep = {v["steps"] for k, v in sums.items() if "trunk_fresh" in k or "trunk_base" in k or "branches" in k}
kl_cfg = [c for c in cfgs if c.get("replay_loss") == "kl"]
import build_lang_data as bld
import c_check as cck
import eval_cycle as evc
import gen_selfgen as gsg

import hparams
HP = hparams.facts()
probe_set_size = len(load_from_disk(evc.PROBE_SET))
ntp_cfg = json.load(open("checkpoints/cycles/branches/trunk_fresh_s42/latest_full/run_config.json", encoding="utf-8"))
ft150_cfg = json.load(open("checkpoints/cycles/branches/trunk_fresh_s42/ft150/run_config.json", encoding="utf-8"))
ep_kept = {src(f"data/cycles/branches/trunk_fresh_s{s}/{a}_replay")["n_blocks"] for s in S for a in ("latest_ep", "base_ep")}
top_p = {src(p)["top_p"] for p in glob.glob("data/cycles/*/*_replay") + glob.glob("data/cycles/branches/*/*_replay")}
# EP 필터로 960블록을 채우려고 실제 생성한 블록 수 (anchor별, 두 줄기)
ep_gen = {a: {src(f"data/cycles/branches/trunk_fresh_s{s}/{a}_replay")["generated_blocks"] for s in S} for a in ("base_ep", "latest_ep")}
# 규칙 상수는 코드에서 읽는다: EP 분류기 (replay_quality.classify), C3 방향 규칙 (c_check.py)
_rq = open("replay_quality.py", encoding="utf-8").read()
EP_RULE = {"nonascii": float(re.search(r"len\(letters\) > ([\d.]+)", _rq).group(1)),
           "sym": float(re.search(r"sym > ([\d.]+) or", _rq).group(1)),
           "code_lines": float(re.search(r"code_lines > ([\d.]+)", _rq).group(1)),
           "stop": float(re.search(r"stop >= ([\d.]+) and", _rq).group(1)),
           "alpha": float(re.search(r"alpha >= ([\d.]+)", _rq).group(1))}
_cc = open("c_check.py", encoding="utf-8").read()
C3_RHO_EN = -float(re.search(r'rho_age_english"\] or 0\) <= -([\d.]+) \+ EPS', _cc).group(1))
C3_RHO_MID = float(re.search(r'rho_age_mid"\] or 0\) >= ([\d.]+) - EPS', _cc).group(1))

# ---------------------------------------------------------------- 검사 목록
thr = CC["thresholds"]
c4 = CC["C4"]
p3 = CC["P3"]["metrics"]
p3_all = [v for d in EN for v in p3[d]["drop_from_M3"].values()]
c3 = CC["C3_pred"]
ft_vs_base = CC["FT150"]["base_full"]
gate = GATE["domains"]
V = lambda *xs: list(xs)

CHECKS = [
    # (이름, 정규식, 기대값 목록, 모드) 모드: "all" = 본문의 모든 등장, "one" = 첫 등장
    ("P5 rho (초록)", r"Spearman ([\d.]+) vs ([\d.]+)", V(CC["P5"]["rho_kl"], CC["P5"]["rho_param_dist"]), "one"),
    ("P3 회복 범위", r"([\d.]+)–([\d.]+) point", V(min(p3_all), max(p3_all)), "all"),
    ("base_full 끝 값 (모든 등장)", r"base_full still ends ([\d.]+)/([\d.]+)%", [enm(B[s]["base_full"]) for s in S], "all"),
    ("P3 회복 범위 (1장)", r"\(by ([\d.]+)–([\d.]+) points\)", V(min(p3_all), max(p3_all)), "all"),
    ("P3 회복 범위 (결론)", r"\(([\d.]+)–([\d.]+) points below the branch start", V(min(p3_all), max(p3_all)), "all"),
    ("흔들림 기준", r"English Wikipedia ([\d.]+), books ([\d.]+), web ([\d.]+), Korean/Japanese/Russian ([\d.]+), Greek ([\d.]+); "
                r"for the averaged metrics used at the ends of the C3 age curve, ([\d.]+) \(English\) and ([\d.]+) \(intermediate\)",
     V(thr["en_wiki"], thr["en_books"], thr["en_web"], max(thr["ko"], thr["ja"], thr["ru"]), thr["el"], thr["en_mean"], thr["mid_mean"]), "one"),
    ("게이트 수치", r"Wikipedia ([+\-\d.]+)/([+\-\d.]+)% vs ([+\-\d.]+)/([+\-\d.]+)%, books ([+\-\d.]+)/([+\-\d.]+)% vs ([+\-\d.]+)/([+\-\d.]+)%, "
               r"web ([+\-\d.]+)/([+\-\d.]+)% vs ([+\-\d.]+)/([+\-\d.]+)%",
     [v for d in EN for v in gate[d]["fresh"] + gate[d]["ft"]], "one"),
    ("B 역전 (사후)", r"\(mean ([+\-\d.]+)% vs ([+\-\d.]+)%\)",
     V(st.mean(v for s in S for v in en(T[f"fresh{s}"]["c3_ru"]).values()), st.mean(v for s in S for v in en(T[f"FT{s}"]["c3_ru"]).values())), "one"),
    ("B 스텝 (120/150)", r"fresh trains (\d+) steps per cycle and FT (\d+)", V(max(steps_rep), max(steps_ft)), "one"),
    ("C1 영어 범위", r"at ([\d.]+)–([\d.]+)% across domains and trunks, versus ([\d.]+)–([\d.]+)% for fresh",
     V(min(v for s in S for v in en(T[f"base{s}"]["c3_ru"]).values()), max(v for s in S for v in en(T[f"base{s}"]["c3_ru"]).values()),
       min(v for s in S for v in en(M3[s]).values()), max(v for s in S for v in en(M3[s]).values())), "one"),
    ("C1-P1 평균 차이", r"\(mean differences ([\-\d.]+), ([\-\d.]+), ([\-\d.]+) points\)",
     [CC["C1-P1"]["metrics"][d]["mean_diff"] for d in EN], "one"),
    ("C1-P2 한국어", r"\(increase from cycle 1 to cycle 3: ([+\-\d.]+)/([+\-\d.]+)% vs ([+\-\d.]+)/([+\-\d.]+)%\)",
     list(CC["C1-P2"]["metrics"]["ko"]["A"].values()) + list(CC["C1-P2"]["metrics"]["ko"]["B"].values()), "one"),
    ("C1-P2 일본어", r"Japanese difference \(([\-\d.]+) points\)", V(CC["C1-P2"]["metrics"]["ja"]["mean_diff"]), "one"),
    ("P1 평균 차이", r"latest-anchor replay \(([\-\d.]+), ([\-\d.]+), ([\-\d.]+) points\)", [CC["P1"]["metrics"][d]["mean_diff"] for d in EN], "one"),
    ("M_3 영어 (회복 문단)", r"\(([\d.]+)/([\d.]+)% above the original model on average\)", [enm(M3[s]) for s in S], "one"),
    ("P3 줄기·영역별", r"by ([\d.]+) and ([\d.]+) points on Wikipedia, ([\d.]+) and ([\d.]+) on books, and ([\d.]+) and ([\d.]+) on web",
     [v for d in EN for v in p3[d]["drop_from_M3"].values()], "one"),
    ("base_full 끝 값", r"base_full still ends ([\d.]+)/([\d.]+)%", [enm(B[s]["base_full"]) for s in S], "one"),
    ("C3 영어 곡선", r"\(([\d.]+) → ([\d.]+) → ([\d.]+) → ([\d.]+)% and ([\d.]+) → ([\d.]+) → ([\d.]+) → ([\d.]+)%; ρ = ([\-\d.]+); ends differ by ([\-\d.]+) points\)",
     c3["per_trunk"]["줄기42"]["english_mean"] + c3["per_trunk"]["줄기43"]["english_mean"]
     + [c3["per_trunk"]["줄기42"]["rho_age_english"], c3["ends_english"]["mean_diff"]], "one"),
    ("P2 일본어 반대", r"went the other way \(([+\-\d.]+) points\)", V(CC["P2"]["metrics"]["ja"]["mean_diff"]), "one"),
    ("C3 중간", r"\(ρ = ([\d.]+) and ([\d.]+); ends differ by ([\d.]+) < ([\d.]+)\)",
     V(c3["per_trunk"]["줄기42"]["rho_age_mid"], c3["per_trunk"]["줄기43"]["rho_age_mid"], c3["ends_mid"]["mean_diff"], thr["mid_mean"]), "one"),
    ("혼합 영어", r"on English \(([\d.]+)/([\d.]+)%\)", [c3["mix"][f"줄기{s}"]["english_mean"] for s in S], "one"),
    ("사후 잔차", r"residuals are ([+\-\d.]+), ([+\-\d.]+) and ([+\-\d.]+) points in trunk 42 and ([+\-\d.]+), ([+\-\d.]+) and ([+\-\d.]+) in trunk 43",
     recompute["residuals"][42] + recompute["residuals"][43], "one"),
    ("5.2 영어 세 값", r"English forgetting was ([\d.]+)/([\d.]+)%, higher than at the branch start M_3 \(([\d.]+)/([\d.]+)%\) and higher than FT-150 \(([\d.]+)/([\d.]+)%\)",
     [enm(B[s]["latest_full"]) for s in S] + [enm(M3[s]) for s in S] + [enm(B[s]["ft150"]) for s in S], "one"),
    ("latest_ep", r"\(latest_ep ([\d.]+)/([\d.]+)%\)", [enm(B[s]["latest_ep"]) for s in S], "one"),
    ("P4 출처", r"\(latest_ep - base_ep: ([+\-\d.]+), ([+\-\d.]+), ([+\-\d.]+) points\)", [CC["P4"]["source_effect"][d]["mean_diff"] for d in EN], "one"),
    ("P4 희석", r"\(latest_full - latest_ep: ([+\-\d.]+), ([+\-\d.]+), ([+\-\d.]+)\)", [CC["P4"]["dilution_effect"][d]["mean_diff"] for d in EN], "one"),
    ("복습 비영어 블록", r"grew from (\d+)–(\d+) of (\d+) at cycle 1 to (\d+)–(\d+) at cycle 3",
     V(min(nonen[1]), max(nonen[1]), max(nblocks), min(nonen[3]), max(nonen[3])), "one"),
    ("base_full vs FT-150 영어", r"English \(([\-\d.]+) to ([\-\d.]+) points vs FT-150\)",
     V(-ft_vs_base["en_wiki"]["mean_diff"], -max(ft_vs_base[d]["mean_diff"] for d in EN)), "one"),
    ("base_full vs FT-150 중간", r"\(Korean ([\-\d.]+), Japanese ([\-\d.]+), Russian ([\-\d.]+) points\)", [-ft_vs_base[l]["mean_diff"] for l in MID], "one"),
    ("그리스어 차이", r"\(difference ([\d.]+) points, margin ([\d.]+)\)", V(abs(ft_vs_base["el"]["mean_diff"]), thr["el"]), "one"),
    ("FT-150 vs M_3 평균", r"\(([+\-\d.]+)% → ([+\-\d.]+)% on average\)", V(mean2(lambda s: enm(M3[s])), mean2(lambda s: enm(B[s]["ft150"]))), "one"),
    ("C4 완화 크기", r"reduced English forgetting by ([\d.]+), ([\d.]+) and ([\d.]+) points",
     [-c4["P1"]["better_than_latest_full"][d]["mean_diff"] for d in EN], "one"),
    ("C4 vs FT-150", r"no difference from FT-150 that meets the practical margin \(([\-\d.]+), ([\-\d.]+), ([\-\d.]+) points\)", [c4["P1"]["vs_ft150"][d]["mean_diff"] for d in EN], "one"),
    ("latest_kl − M_3", r"\(([+\-\d.]+) and ([+\-\d.]+) points; trunk 43 web was the exception, ([\d.]+) vs ([\d.]+)%\)",
     [enm(B[s]["latest_kl"]) - enm(M3[s]) for s in S] + [en(B[43]["latest_kl"])["en_web"], en(M3[43])["en_web"]], "one"),
    ("완화 비율", r"was ([\d.]+), ([\d.]+) and ([\d.]+) by domain", [c4["P1"]["mitigation_ratio"][d]["pooled"] for d in EN], "one"),
    ("base KL vs NTP", r"base_kl forgot English ([\d.]+)% on average vs ([\d.]+)% for base_full",
     V(mean2(lambda s: enm(B[s]["base_kl"])), mean2(lambda s: enm(B[s]["base_full"]))), "one"),
    ("책 기준 (C4-P2)", r"below the ([\d.]+) margin", V(thr["en_books"]), "one"),
    ("KL 이득 비", r"about ([\d.]+) times on average \(([\d.]+) vs ([\d.]+) points\), with a wide spread \(([\d.]+) and ([\d.]+) times by trunk; ([\d.]+)–([\d.]+) by domain\)",
     V(mean2(lambda s: gain[s][0]) / mean2(lambda s: gain[s][1]), mean2(lambda s: gain[s][0]), mean2(lambda s: gain[s][1]),
       gain[42][0] / gain[42][1], gain[43][0] / gain[43][1], min(gain_dom.values()), max(gain_dom.values())), "one"),
    ("C4 중간 보고", r"Japanese \(([\-\d.]+) points\) and Russian \(([\-\d.]+)\)",
     V(c4["report_only"]["latest_kl_vs_latest_full"]["ja"]["mean_diff"], c4["report_only"]["latest_kl_vs_latest_full"]["ru"]["mean_diff"]), "one"),
    ("grad_norm 평균", r"latest ([\d.]+)/([\d.]+) vs ([\d.]+)/([\d.]+)\)",
     V(gn[(42, "latest_full")], gn[(43, "latest_full")], gn[(42, "latest_kl")], gn[(43, "latest_kl")]), "one"),
    ("복습 KL 로그", r"about ([\d.]+) nats/token for the latest teacher, ([\d.]+) for the base teacher", V(rkl["latest_kl"], rkl["base_kl"]), "one"),
    ("P5 본문", r"n = (\d+)\), English forgetting was strongly associated with probe KL to the original model \(Spearman ρ = ([\d.]+)\) and less strongly "
               r"with relative parameter distance \(ρ = ([\d.]+)\)", V(CC["P5"]["n"], CC["P5"]["rho_kl"], CC["P5"]["rho_param_dist"]), "one"),
    ("P5 C3 포함", r"Adding the C3 branches gives ([\d.]+) vs ([\d.]+) \(n = (\d+)\)",
     V(CC["P5"]["with_C3_reference"]["rho_kl"], CC["P5"]["with_C3_reference"]["rho_param_dist"], CC["P5"]["with_C3_reference"]["n"]), "one"),
    ("표 1 P5", r"Pre-specified descriptive comparison: ([\d.]+) vs ([\d.]+) \|", V(CC["P5"]["rho_kl"], CC["P5"]["rho_param_dist"]), "one"),
    # 3.4 규칙 (코드 상수와 대조)
    ("C3 방향 규칙 rho", r"is at most ([\-\d.]+) \(forgetting decreases with age\).*?with ρ at least ([\d.]+) \(forgetting increases with age\)",
     V(C3_RHO_EN, C3_RHO_MID), "one"),
    # EP 필터 규칙과 생성 비용 (replay_quality.classify, source.json)
    ("EP 필터 규칙", r"at most (\d+)% of its letters are non-ASCII, it is not flagged as code \(code-symbol density above (\d+)% or more than "
                r"(\d+)% code-like lines\), at least (\d+)% of its words are common English function words, and letters make up at least (\d+)%",
     V(*(EP_RULE[k] * 100 for k in ("nonascii", "sym", "code_lines", "stop", "alpha"))), "one"),
    ("EP 생성 블록", r"To keep (\d+) blocks, the EP arms generated ([\d,]+) \(base\) and ([\d,]+) \(latest\) blocks",
     V(max(ep_kept), max(ep_gen["base_ep"]), max(ep_gen["latest_ep"])), "one"),
    # 외부 논문 인용 수치: 결과 파일과 무관, 저자가 원문에서 확인한 값 (2026-09-29 전달). 본문이 바뀌지 않았는지만 확인
    ("외부 인용: ComposeCL 과제 수", r"over (\d+) question-answering tasks", V(100), "one"),
    ("외부 인용: ComposeCL 최종 기억률", r"average final retention rose from ([\d.]+)% to ([\d.]+)%", V(1.2, 34.9), "one"),
    ("외부 인용: Chen App. A.5 Pearson", r"report a Pearson correlation of ([\d.]+) between KL and forgetting", V(0.52), "one"),
    ("외부 인용: Kotha & Liang 설정 크기", r"in a controlled (\d+)M-parameter setting", V(150), "one"),
    # 한계
    ("학습 글 수", r"from (\d+)–(\d+) articles per language", V(min(r["train_docs"] for r in BUILD.values()), max(r["train_docs"] for r in BUILD.values())), "one"),
    ("오염 규칙 제외율", r"removed (\d+)% of candidate held-out blocks for Korean and (\d+)% for Greek",
     V(BUILD["ko"]["drop_rate"] * 100, BUILD["el"]["drop_rate"] * 100), "one"),
    ("글자 수 비", r"Russian about ([\d.]+) times more characters than Korean", V(BUILD["ru"]["train_stats"]["chars"] / BUILD["ko"]["train_stats"]["chars"]), "one"),
    ("cycle당 토큰", r"about ([\d.]+)M tokens per cycle", V(bld.N_TRAIN * bld.BLOCK_SIZE / 1e6), "all"),
    ("옛 한국어 겹침", r"shared ([\d.]+)% of its 13-grams", V(old_ko_overlap), "one"),
    ("연산량 25%", r"fresh 150 \((\d+)% more compute\)", V((max(steps_rep) / max(steps_ft) - 1) * 100), "one"),
    ("그리스어 노출 25%", r"\((\d+)% more Greek exposure\)", V(960 / 3840 * 100), "one"),
    ("코드 비율", r"contains code \(about (\d+)%\)", V(code_n / gen_n * 100), "one", [0.5]),
    ("망가진 블록", r"repetition ≥ ([\d.]+), about (\d+)%\)", V(0.5, sum(broken) / len(broken) * 100), "one", [0.05, 0.5]),
    ("생성 배치", r"generation used batch (\d+) and C-stage batch (\d+)", V(max(gen_batch["B"]), max(gen_batch["C"])), "one"),
    # KL 바닥값: results/cycles/kl_floor.json (measure_kl_floor.py, 학생 = 교사 가중치, bf16 학습 forward, 복습 960블록 x 2줄기)
    ("KL 바닥값 (latest 교사 ~ base 교사)", r"replay text \(about ([\d.]+)–([\d.]+) nats/token\)", V(KLF["latest_mean"], KLF["base_mean"]), "one"),
    # 부록 체크리스트
    ("부록 base KL vs NTP", r"\(7\.1 vs ([\d.]+)%, Wikipedia", V(mean2(lambda s: enm(B[s]["base_full"]))), "one"),
    ("부록 base_kl 7.1", r"\(([\d.]+) vs [\d.]+%, Wikipedia and web meet the margin, books not\)", V(mean2(lambda s: enm(B[s]["base_kl"]))), "one"),
    ("부록 완화 비율", r"ratios ([\d.]+) / ([\d.]+) / ([\d.]+)", [c4["P1"]["mitigation_ratio"][d]["pooled"] for d in EN], "one"),
    ("부록 이득 비", r"about ([\d.]+) times larger gain", V(mean2(lambda s: gain[s][0]) / mean2(lambda s: gain[s][1])), "one"),
    ("부록 이득 퍼짐", r"spread of ([\d.]+)–([\d.]+) by trunk", V(gain[42][0] / gain[42][1], gain[43][0] / gain[43][1]), "one"),
    ("부록 P5", r"\(ρ = ([\d.]+) vs ([\d.]+), n = (\d+), pre-specified set\)", V(CC["P5"]["rho_kl"], CC["P5"]["rho_param_dist"], CC["P5"]["n"]), "one"),
    # probe 겹침 (5.4, 한계): probe 블록은 영어 위키 평가 세트의 일부 → 영역별 rho
    ("probe 겹침 (5.4)", r"English Wikipedia evaluation set \((\d+) of its (\d+) blocks\)", V(len(evc.PROBE_IDX), probe_set_size), "one"),
    ("probe 겹침 (한계)", r"English Wikipedia evaluation set \((\d+) of (\d+) blocks\)", V(len(evc.PROBE_IDX), probe_set_size), "one"),
    ("P5 영역별 rho", r"books ρ = ([\d.]+), web ρ = ([\d.]+); Wikipedia ρ = ([\d.]+)",
     V(CC["P5"]["per_domain"]["en_books"]["kl"], CC["P5"]["per_domain"]["en_web"]["kl"], CC["P5"]["per_domain"]["en_wiki"]["kl"]), "one"),
    # 부록 A 하이퍼파라미터 표 (hparams.facts: TrainingArguments·run_config·source.json·코드 상수)
    ("표 3 β", r"\| Adam β1, β2 \| ([\d.]+), ([\d.]+) \|", V(HP["beta1"], HP["beta2"]), "one"),
    ("표 3 ε", r"\| Adam ε \| ([\d.]+e-\d+) \|", V(HP["eps"]), "one", [1e-12]),
    ("표 3 weight decay", r"\| Weight decay \| ([\d.]+) \|", V(HP["wd"]), "one"),
    ("표 3 clipping", r"\| Gradient clipping \(max norm\) \| ([\d.]+) \|", V(HP["clip"]), "one"),
    ("표 3 LR", r"\| Learning rate \| ([\d.]+e-\d+), constant after a (\d+)-step linear warmup \|", V(HP["lr"], HP["warmup"]), "one", [1e-12, 0.5]),
    ("표 3 배치", r"\| Batch per update \| (\d+) blocks of (\d+) tokens \(NTP (\d+) × (\d+), KL (\d+) × (\d+)\) \|",
     V(HP["ntp_micro"] * HP["ntp_accum"], HP["block"], HP["ntp_micro"], HP["ntp_accum"], HP["kl_micro"], HP["kl_accum"]), "one"),
    ("표 3 스텝", r"\| Steps per cycle \| (\d+) with replay \(([\d,]+) new \+ (\d+) replay blocks\), (\d+) without \(FT\) \|",
     V(HP["steps_replay"], HP["new_n"], HP["replay_n"], HP["steps_ft"]), "one"),
    ("표 3 시드", r"\| Seeds \| (\d+), (\d+) \(training seed = trunk seed\) \|", HP["seeds"], "one"),
    ("표 3 temperature", r"\| Temperature \| ([\d.]+) \|", V(next(iter(HP["temperature"]))), "one"),
    ("표 3 top-p", r"\| Top-p \| ([\d.]+) \(no top-k; padding ids blocked\) \|", V(next(iter(HP["top_p"]))), "one"),
    ("표 3 생성 길이", r"\| Length \| (\d+) tokens per block \|", V(next(iter(HP["gen_len"]))), "one"),
    ("표 3 생성 배치", r"\| Batch \| (\d+) \(B stage\), (\d+) \(C stage\) \|", V(next(iter(HP["gen_batch_B"])), next(iter(HP["gen_batch_C"]))), "one"),
    ("표 3 블록 길이", r"\| Block length \| (\d+) tokens \((\d+) prediction positions\) \|", V(HP["block"], HP["block"] - 1), "one"),
    ("표 3 held-out", r"\| Held-out blocks \| (\d+) per set", V(next(iter(HP["eval_blocks"]))), "one"),
    ("표 3 probe", r"\| Probe for probe KL \| (\d+) blocks, every (\d+)th block", V(HP["probe_n"], HP["probe_step"]), "one"),
    ("표 3 평가 배치", r"\| Precision \| fp16 weights, batch (\d+) \|", V(HP["eval_batch"]), "one"),
    ("표 3 글 상한", r"\| Article cap \| (\d+) blocks \(training\), (\d+) blocks \(held-out\) \|", V(HP["train_cap"], HP["held_cap"]), "one"),
    ("표 3 오염 규칙", r"more than (\d+)% of their (\d+)-grams in the training data", V(HP["ngram_rule"] * 100, 13), "one"),
    ("C3 ρ = −1 (부록)", r"\(ρ = ([\-\d]+), ends meet the margin\)", V(c3["per_trunk"]["줄기43"]["rho_age_english"]), "one"),
    ("B 스텝 (부록)", r"without the (\d+) vs (\d+) step confound", V(max(steps_rep), max(steps_ft)), "one"),
    # 설정 상수 (추가)
    ("B 생성 블록", r"generate (\d+) blocks from the current checkpoint", V(max(nblocks)), "one"),
    ("EP 통과 블록", r"EP arms generate until (\d+) blocks pass the filter", V(max(ep_kept)), "one"),
    ("FT-150 반복", r"with the (\d+) replay slots filled by a random repeat of (\d+) Greek training blocks",
     V(960, ft150_cfg["n_repeat_blocks"]), "one"),
    ("업데이트 블록", r"every update contains the same (\d+) blocks", V(ntp_cfg["batch_size"] * ntp_cfg["grad_accum"]), "one"),
    ("글당 블록 (한계)", r"\(at most (\d+) blocks each\)", V(bld.TRAIN_PER_DOC), "one"),
    ("블록 토큰 (한계)", r"The same (\d+) tokens carry", V(bld.BLOCK_SIZE), "one"),
    ("B 스텝 (한계)", r"FT trains (\d+) steps per cycle and fresh (\d+)", V(max(steps_ft), max(steps_rep)), "one"),
    ("KL·NTP 배치 (한계)", r"micro-batch (\d+) × (\d+) accumulation versus (\d+) × (\d+) for NTP",
     V(kl_cfg[0]["micro_batch"], kl_cfg[0]["grad_accum"], ntp_cfg["batch_size"], ntp_cfg["grad_accum"]), "one"),
    ("top-p (관련 연구)", r"\(p = ([\d.]+), as here\)", V(max(top_p)), "one"),
    ("top-p (한계)", r"sampled with top-p ([\d.]+), so it follows", V(max(top_p)), "one"),
    ("기준 하한 (3.4)", r"floor ([\d.]+) percentage points", V(cck.FLOOR), "one"),
    ("기준 하한 (한계)", r"with a ([\d.]+)-point floor", V(cck.FLOOR), "one"),
    ("cycle당 토큰 (3.1)", r"\(about ([\d.]+)M tokens\)", V(bld.N_TRAIN * bld.BLOCK_SIZE / 1e6), "one"),
    # 설정 상수 (실제 설정 파일·코드에서)
    ("새 블록 수", r"trains on ([\d,]+) new blocks of (\d+) tokens", V(bld.N_TRAIN, bld.BLOCK_SIZE), "one"),
    ("글당 블록 상한", r"capping each article at (\d+) blocks", V(bld.TRAIN_PER_DOC), "one"),
    ("학습률·워밍업·배치", r"learning rate of ([\d.e\-]+) after a (\d+)-step linear warmup, (\d+) blocks per update",
     V(cfgs[0]["lr"], cfgs[0]["warmup_steps"], cfgs[0]["batch_size"] * 8), "one"),
    ("복습 비율", r"(\d+) replay blocks are added \((\d+)% of ([\d,]+) blocks, (\d+) steps\)", V(960, 20, 4800, max(steps_rep)), "one"),
    ("FT 스텝", r"without replay \(FT\) a cycle has (\d+) steps", V(max(steps_ft)), "one"),
    ("샘플링", r"samples (\d+) tokens starting from a single EOS token \(unconditional\), with temperature ([\d.]+) and top-p ([\d.]+)",
     V(gsg.BLOCK_SIZE, 1.0, 0.95), "one"),
    ("held-out 크기", r"C4 web text \(Raffel et al\., 2020\); (\d+) blocks each", V(400), "one"),
    ("held-out 규칙", r"keep at most (\d+) blocks per article, and drop any block whose (\d+)-grams overlap the training data by more than (\d+)%",
     V(bld.HELDOUT_PER_DOC, 13, bld.MAX_BLOCK_OVERLAP * 100), "one"),
    ("probe 블록", r"on (\d+) fixed English Wikipedia blocks", V(len(evc.PROBE_IDX)), "one"),
    ("KL 배치", r"micro-batch (\d+) × (\d+) accumulation\) and KL \((\d+) × (\d+)\)",
     V(4, 8, kl_cfg[0]["micro_batch"], kl_cfg[0]["grad_accum"]), "one"),
    ("예측 위치", r"same (\d+) prediction positions per block", V(bld.BLOCK_SIZE - 1), "one"),
    ("혼합 크기", r"a 50/50 mix \((\d+) base \+ (\d+) latest\)", V(480, 480), "one"),
    ("표 2 복습 수", r"re-use exactly the (\d+) replay blocks", V(960), "one"),
]
if SCRATCH_EQ:
    eq = json.load(open(SCRATCH_EQ[0], encoding="utf-8"))
    CHECKS.append(("greedy 일치", r"matched on ([\d,]+)/([\d,]+) tokens",
                   V(eq["greedy_fp16"]["tokens_equal"], eq["greedy_fp16"]["tokens_total"]), "one"))

# 표 2 행
TABLE2 = {"FT-150": "ft150", "latest_full": "latest_full", "latest_ep": "latest_ep", "latest_kl": "latest_kl",
          "base_full": "base_full", "base_ep": "base_ep", "base_kl": "base_kl"}
for label, arm in TABLE2.items():
    exp = [enm(B[s][arm]) for s in S] + [midm(B[s][arm], M3[s]) for s in S] + [pct(B[s][arm]["ppl_el"], M3[s]["ppl_el"]) for s in S]
    CHECKS.append((f"표 2 {label}", r"\| " + re.escape(label) + r" \|[^|]*\|[^|]*\| ([\d.]+) / ([\d.]+) \| ([\d.]+) / ([\d.]+) \| ([\-\d.]+) / ([\-\d.]+) \|", exp, "one"))
CHECKS.append(("표 2 M_3", r"\| M_3 \(branch start\) \|[^|]*\|[^|]*\| ([\d.]+) / ([\d.]+) \|", [enm(M3[s]) for s in S], "one"))

# 판정 문구 ("certain" → "meets the practical margin": c_check.json의 certain 필드와 같은 뜻)
P4_WORDING = (CC["P4"]["verdict"].startswith("자기 강화 지지") and all(CC["P4"]["source_effect"][d]["certain"] for d in EN)
              and not any(CC["P4"]["dilution_effect"][d]["certain"] for d in EN))
VERDICTS = [
    ("표 1 Gate", r"\| Gate \|[^|]*\| Passed in all three domains \|", GATE["verdict"].startswith("통과")),
    ("표 1 C1-P1", r"\| C1-P1 \|[^|]*\| Supported \(all three domains\) \|", CC["C1-P1"]["verdict"] == "지지"),
    ("표 1 C1-P2", r"\| C1-P2 \|[^|]*\| Partial \(Korean yes; Japanese below margin\) \|",
     CC["C1-P2"]["verdict"] == "부분 지지" and not CC["C1-P2"]["metrics"]["ja"]["certain"]),
    ("표 1 P1", r"\| P1 \|[^|]*\| Supported \(all three domains\) \|", CC["P1"]["verdict"] == "지지"),
    ("표 1 P2", r"\| P2 \|[^|]*\| Mixed \(Korean, Russian yes; Japanese opposite\) \|",
     CC["P2"]["verdict"] == "혼재" and CC["P2"]["metrics"]["ja"]["outcome"] == "반대 방향 확실"),
    ("표 1 P3", r"\| P3 \|[^|]*\| Supported \(all three domains\) \|", CC["P3"]["verdict"] == "지지"),
    ("표 1 P4", r"\| P4 \|[^|]*\| Source effect meets margin; dilution effect below margin \|", P4_WORDING),
    ("5.2 P4 문구", r"points\) met the practical margin in every domain, while the dilution effect \(latest_full - latest_ep: [^)]*\) was below "
                  r"the margin in every domain", P4_WORDING),
    ("5.2 FT-150 대비 margin", r"\(differences to FT-150 meet the practical margin in all three domains\)",
     all(CC["FT150"]["latest_full"][d]["certain"] for d in EN)),
    ("표 1 P5", r"\| P5 \|[^|]*\| Pre-specified descriptive comparison: ",
     CC["P5"]["verdict"] == "지지" and CC["P5"]["rho_kl"] > CC["P5"]["rho_param_dist"]),
    ("EP 생성 블록 수가 두 줄기에서 같음", r"the EP arms generated", all(len(v) == 1 for v in ep_gen.values())),
    ("표 1 C3", r"\| C3 \|[^|]*\| Partial: English only \|", c3["verdict"].startswith("부분 지지 (영어만)")),
    ("표 1 C4-P1", r"\| C4-P1 \|[^|]*\| Category A in all three domains \|", set(c4["P1"]["per_domain"].values()) == {"A"}),
    ("표 1 C4-P2", r"\| C4-P2 \|[^|]*\| Supported \|", c4["P2"]["verdict"] == "지지"),
    ("C4-P2 위키·웹 margin 충족, 책 아님", r"\(Wikipedia and web meet the practical margin, books is below", [c4["P2"]["metrics"][d]["certain"] for d in EN] == [True, False, True]),
    ("재시도 없음 (C3·C4)", r"none occurred during the C3 and C4 runs", not os.path.exists(f"{R}/native_crash_retries.jsonl")),
    ("probe가 영어 위키 평가 세트에서 나옴 (코드)", r"The probe blocks are part of the English Wikipedia evaluation set",
     evc.PROBE_SET == evc.SETS["en_wiki"]),
    ("표 3 옵티마이저 (코드)", r"\| Optimizer \| AdamW 8-bit \(bitsandbytes, `?adamw_bnb_8bit`?\), re-initialized every cycle \|",
     HP["optim"] == "adamw_bnb_8bit"),
    ("표 3 정밀도 (코드)", r"\| Precision \| fp32 weights with bf16 autocast, gradient checkpointing; weights saved in fp16 after each cycle \|",
     HP["bf16"] and HP["gc"]),
    ("태그 정의 (사전 지정)", r"\[pre-specified\]\*?\*? = decision rule fixed in the lab notes before the result was seen; not publicly registered "
                         r"with a timestamp", True),
    # 관련 연구 문구 (원문 확인 내용, 2026-09-29): OPR = 보상으로 거른 (프롬프트, 응답) 쌍을 일반 SFT로 replay, "관련 관찰"로만 연결
    ("OPR 서술 (보상 필터, 교사·보조 손실 없음)", r"filter them by task reward, and replay the surviving \(prompt, model response\) pairs as "
                                          r"ordinary SFT examples, with no teacher and no auxiliary loss", True),
    ("OPR과의 연결은 '관련 관찰'", r"a related observation to our result for unfiltered latest-checkpoint replay", True),
    ("저장소 주소", r"available at https://github\.com/DoHyunwooo/replay-anchor-provenance", True),
    ("공백 블록 4개", r"four identical whitespace-only blocks",
     sum(n for v in dup.values() for n, ws in v if ws) == 4 and sum(len(v) for v in dup.values()) == 1),
]


# ---------------------------------------------------------------- 대조
def norm_text(t):
    t = t.replace("−", "-")
    return re.sub(r"\s+", " ", t)


def num(sv):
    return float(sv.replace(",", "").replace("+", ""))


def decimals(sv):
    sv = sv.replace(",", "")
    return len(sv.split(".")[1]) if "." in sv and "e" not in sv else 0


def tex_to_text(t):
    """md2tex.py가 만든 LaTeX를 마크다운과 같은 글로 되돌린다 (검사 정규식을 그대로 쓰기 위해)."""
    t = t.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    t = re.sub(r"(?<!\\)%.*", "", t)
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if re.match(r"^\\(begin|end)\{(figure|table|tabularx?|enumerate|itemize|abstract)\}", s) or \
                re.match(r"^\\(centering|small|toprule|midrule|bottomrule|includegraphics|label|maketitle|bibliography|bibliographystyle|appendix)", s):
            continue
        if "&" in s and s.endswith(r"\\") and not s.startswith(r"\author"):
            cells = [c.strip() for c in re.split(r"(?<!\\)&", s[:-2])]
            s = "| " + " | ".join(cells) + " |"
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"\\cite[pt](\[[^\]]*\])?\{[^}]*\}", "CITE", t)
    t = t.replace(r"\paramdist{}", "‖θ − θ0‖ / ‖θ0‖")
    t = re.sub(r"\$M_\{?(\w)\}?\$", r"M_\1", t)
    for cmd in ("textbf", "emph", "texttt", "url", r"section\*?", "subsection", "caption", "title"):
        t = re.sub(r"\\" + cmd + r"\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", t)
    t = re.sub(r"\\item\s*", "- ", t)
    t = (t.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_").replace("``", '"').replace("''", '"')
         .replace(r"\#", "#").replace("~", " "))
    t = re.sub(r"\\noindent\{\\small ((?:[^{}]|\{[^{}]*\})*)\}", r"\1", t)
    t = re.sub(r"\{\\sloppy (.*?)\\par\}", r"\1", t, flags=re.S)       # md2tex의 SLOPPY_PARAS 문단
    t = re.sub(r"\$\\(Vert|rho)\$", lambda m: {"Vert": "‖", "rho": "ρ"}[m.group(1)], t)
    return t


import argparse
ap = argparse.ArgumentParser(description="초안 수치 대조")
ap.add_argument("--source", default=DRAFT, help="PAPER_DRAFT.md (기본) 또는 paper/main.tex")
ap.add_argument("--release", action="store_true",
                help="공개 직전 검사: 내부 메모(원문 재확인 메모, [GitHub URL to be added], 서지 확인 태그, 미완성 저자 목록)가 남아 있으면 실패")
cli = ap.parse_args()
IS_TEX = cli.source.endswith(".tex")
raw = open(cli.source, encoding="utf-8").read()
def citation_checks(tex_raw):
    """본문 인용과 references.bib 대조: 인용 키 = bib 키 (양쪽 빠짐 없음), 변환 안 된 인용 문구 없음,
    인용 문구의 저자 성 = bib 첫 저자 성, main.bbl에 모든 인용 키."""
    import md2tex
    bib = open("paper/references.bib", encoding="utf-8").read()
    entries = dict(re.findall(r"@\w+\{([\w-]+),(.*?)\n\}", bib, re.S))
    first_author = {k: re.search(r"author\s*=\s*\{([^,}\s]+)", v).group(1) for k, v in entries.items()}
    bib_year = {k: re.search(r"year\s*=\s*\{(\d{4})\}", v).group(1) for k, v in entries.items()}
    cited = set(k.strip() for m in re.findall(r"\\cite[pt](?:\[[^\]]*\])?\{([^}]*)\}", tex_raw) for k in m.split(","))
    body_tex = tex_raw.split(r"\begin{document}", 1)[1]
    leftover_cites = re.findall(r"[A-Z][a-z]+(?: et al\.| & [A-Z][a-z]+),? \(?\d{4}", re.sub(r"\\cite[pt](\[[^\]]*\])?\{[^}]*\}", "", body_tex))
    surname_bad, year_bad = [], []
    for md_phrase, tex in md2tex.CITES:
        key = re.search(r"\{([\w-]+)\}$", tex).group(1)
        name = re.match(r"\(?([A-Z][a-z]+)", md_phrase).group(1)
        if key not in first_author or first_author[key] != name:
            surname_bad.append((md_phrase, key, first_author.get(key)))
        year = re.search(r"\b(19|20)\d\d\b", md_phrase).group(0)
        if bib_year.get(key) != year:
            year_bad.append((md_phrase, key, bib_year.get(key)))
    bbl = open("paper/main.bbl", encoding="utf-8").read() if os.path.exists("paper/main.bbl") else ""
    bbl_keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([\w-]+)\}", bbl, re.S))
    return [
        (f"인용 키 = bib 키 (본문 {len(cited)}개, bib {len(entries)}개; 인용 안 된 bib {sorted(set(entries) - cited)}, "
         f"bib에 없는 인용 {sorted(cited - set(entries))})", cited == set(entries)),
        (f"변환 안 된 인용 문구 없음 {leftover_cites[:3]}", not leftover_cites),
        (f"인용 문구의 저자 성 = bib 첫 저자 {surname_bad}", not surname_bad),
        (f"인용 문구의 연도 = bib 연도 {year_bad}", not year_bad),
        (f"main.bbl에 모든 인용 키 (빠진 것 {sorted(cited - bbl_keys)})", cited <= bbl_keys),
    ]


CITE_RESULTS = []
if IS_TEX:
    CITE_RESULTS = citation_checks(raw)
    body = norm_text(tex_to_text(raw))
    # arXiv 판에는 부록 체크리스트가 없다 → 부록 검사는 건너뜀 (내부 문서 internal/CLAIM_CHECKLIST.md)
    CHECKS = [c for c in CHECKS if "부록" not in c[0]]
    # 인용이 CITE로 바뀌므로 인용이 들어간 검사 문구를 맞춘다
    CHECKS = [(c[0], c[1].replace(r"\(Raffel et al\., 2020\)", "CITE"), *c[2:]) for c in CHECKS]
else:
    text = norm_text(raw)
    # 참고문헌 목록만 빼고 본문 + 부록 체크리스트를 대조한다
    pre, rest = text.split("## References", 1)
    body = pre + "## Appendix" + rest.split("## Appendix", 1)[1]
covered = []
bad, ok, missing = [], 0, []
for chk in CHECKS:
    name, pat, exp, mode = chk[:4]
    tols = chk[4] if len(chk) > 4 else None     # "about" 값은 검사마다 허용 범위를 따로 준다
    ms = list(re.finditer(pat, body))
    if not ms:
        missing.append(name)
        continue
    for m in (ms if mode == "all" else ms[:1]):
        covered.append(m.span())
        for i, (sv, e) in enumerate(zip(m.groups(), exp)):
            v = num(sv)
            tol = tols[i] if tols else 0.5 * 10 ** -decimals(sv) + 1e-9
            good = abs(v - e) <= tol
            ok += good
            if not good:
                bad.append(f"{name}: 본문 {sv} vs 계산 {e:.4f} (허용 ±{tol:g})")
vbad = []
# 용어 통일 (외부 리뷰 C3): "certain"/"noise threshold"/"threshold" 대신 "meets the practical margin"/"practical margin"/"margin"
old_terms = sorted({m.group(0).lower() for m in re.finditer(r"\b(certain(ly)?|noise thresholds?|thresholds?)\b", body, re.I)})
if old_terms:
    vbad.append(f"옛 용어가 남아 있음 (practical margin으로 바꿀 것): {old_terms}")
else:
    ok += 1
# 쓰면 안 되는 문구 (논문 본문만; 체크리스트는 금지 문구를 인용하므로 뺀다)와 다른 논문의 arXiv 번호.
# Kotha & Liang은 arXiv:2603.04964. WRONG_ARXIV는 헷갈리기 쉬운 다른 논문 번호 (정규식으로만 적어 저장소 검색에 걸리지 않게 함)
WRONG_ARXIV = r"2603\.16177"
paper_body = body.split("## Appendix: Claim checklist")[0]
FORBIDDEN = [r"high-confidence", r"confidence filter", r"filtered by confidence", r"1\.87\s*×", r"up to 1\.87", WRONG_ARXIV]
forbidden_hits = [p for p in FORBIDDEN if re.search(p, paper_body, re.I)]
_bib_now = open("paper/references.bib", encoding="utf-8").read()
kotha_ok = "2603.04964" in re.search(r"@\w+\{kotha2026,(.*?)\n\}", _bib_now, re.S).group(1) and not re.search(WRONG_ARXIV, _bib_now)
if forbidden_hits or not kotha_ok:
    vbad.append(f"금지 문구·번호: 본문 {forbidden_hits}, kotha2026 = arXiv:2603.04964 {kotha_ok}")
else:
    ok += 1

# 공개 직전 검사 (--release에서만 실패, 평소에는 목록만 보고)
RELEASE_MEMOS = [r"All summaries: recheck", r"\[GitHub URL to be added\]", r"recheck all against the originals", r"to recheck",
                 r"\[authors unconfirmed\]", r"\[venue to fill\]", r"to be added", r"to fill\b", r"\bTODO\b"]
release_hits, _seen = [], []
for p in RELEASE_MEMOS:
    for m in re.finditer(p, raw):
        if any(a < m.end() and m.start() < b for a, b in _seen):     # 이미 다른 패턴으로 잡힌 자리 (예: "[GitHub URL to be added]")
            continue
        _seen.append(m.span())
        release_hits.append(f"본문: '{m.group(0)}' … {raw[max(0, m.start() - 30):m.end() + 30].replace(chr(10), ' ')}")
if IS_TEX:
    # 서지: 주석(% …)은 PDF에 안 나오므로 빼고, 실제 필드에서 미완성 저자 목록 ("and others")과 메모를 찾는다
    bib_live = "\n".join(l for l in open("paper/references.bib", encoding="utf-8").read().split("\n") if not l.lstrip().startswith("%"))
    for m in re.finditer(r"@\w+\{([\w-]+),(.*?)\n\}", bib_live, re.S):
        if re.search(r"author\s*=\s*\{[^}]*\band others\b", m.group(2)):
            release_hits.append(f"references.bib: {m.group(1)} 저자 목록 미완성 ('and others')")
        for p in RELEASE_MEMOS:
            if re.search(p, m.group(2)):
                release_hits.append(f"references.bib: {m.group(1)} 메모 '{p}'")
    # 서지 주석의 [확인 필요]: 아직 확인하지 않은 필드가 있다는 표시 (PDF에 찍히든 아니든 공개 전에 정리)
    bib_all = open("paper/references.bib", encoding="utf-8").read()
    for m in re.finditer(r"((?:^%.*\n)+)@\w+\{([\w-]+),", bib_all, re.M):
        for note in re.findall(r"\[확인 필요\]\s*(.*)", m.group(1)):
            release_hits.append(f"references.bib: {m.group(2)} [확인 필요] {note.strip()}")
if cli.release and release_hits:
    vbad.append(f"공개 직전 검사 실패: 내부 메모·미완성 항목 {len(release_hits)}개 (아래 [공개 직전 검사] 참고)")

for name, cond in CITE_RESULTS:
    ok += bool(cond)
    if not cond:
        vbad.append(f"인용 검사 실패: {name}")
for name, pat, cond in VERDICTS:
    m = re.search(pat, body)
    if not m:
        missing.append(name)
    else:
        covered.append(m.span())
        ok += bool(cond)
        if not cond:
            vbad.append(f"{name}: 본문 문구와 c_check/gate 판정이 다름")

# 검증 안 된 수치 (절·그림·표·연도·이름 번호 제외)
# 결과 수치가 아닌 숫자 (구조·이름·날짜·형식): 이 문맥 안의 숫자는 보고하지 않는다
# (긴 패턴을 먼저: 날짜·"seeds 42, 43"·"age 0 to 3"이 짧은 패턴에 먼저 걸려 쪼개지지 않게)
IGNORE = re.compile(r"(\d{4}-\d\d-\d\d|seeds? 42, 43|seeds? 42 and 43|age 0 to 3|0 = M_3|3 = original model|§\d[\d.–]*|Figs?\. [\d, a-b]+|Fig\. \d[ab]?|Table \d|"
                    r"Eq\. \d|M_\d|\bc\d\b|cycle \d|cycles 1→3|\bP\d|C\d(-P\d)?|H\d|\b(19|20)\d\d\b|Pythia-410M|410M|"
                    r"arXiv[: ]\d+\.\d+|PMLR \d+|trunk 4[23]|42 / 43|2×2|50/50|top-p 1\.0|0 to 3|C1–C4|C2–C4|"
                    r"Claim \d[b]?|## \d\.|### \d\.\d|\s\d\. \*\*|fp16|bf16|PG-19|FT-150|FT150|8-bit|θ0|4-gram|chapter 9|"
                    r"§5\.1–5\.3|\d-step|n = \d+|App\. A\.\d|age [0-3]\b)")
NOTE_ONLY = {}   # 결과 파일로 확인할 수 없는 값 (현재 없음: KL 바닥값은 kl_floor.json으로 검증)
leftover, note_only = [], []
for m in re.finditer(r"[+\-]?\d[\d,]*(?:\.\d+)?(?:e-\d+)?", body):
    s0, e0 = m.span()
    if any(a <= s0 and e0 <= b for a, b in covered):
        continue
    ctx = body[max(0, s0 - 25):min(len(body), e0 + 25)]
    near = body[max(0, s0 - 20):min(len(body), e0 + 20)]
    if any(a.start() <= (s0 - max(0, s0 - 20)) < a.end() and a.end() > a.start() for a in IGNORE.finditer(near)):
        continue
    if m.group() in NOTE_ONLY:
        note_only.append(f"'{m.group()}': {NOTE_ONLY[m.group()]}")
        continue
    leftover.append(f"'{m.group()}' … {ctx}")

rep = [f"{cli.source} 수치 대조 ({len(CHECKS)}개 수치 검사 + {len(VERDICTS)}개 판정 검사)"
       + (" — LaTeX 본문, 부록 검사 제외" if IS_TEX else ""),
       f"  일치 {ok}개, 불일치 {len(bad) + len(vbad)}개, 본문에서 못 찾은 검사 {len(missing)}개, 검증 안 된 숫자 {len(leftover)}개", ""]
rep += ["[불일치]"] + (bad + vbad or ["  없음"]) + [""]
if CITE_RESULTS:
    rep += ["[인용 검사 (LaTeX)]"] + [f"  {'O' if c else 'X'} {n}" for n, c in CITE_RESULTS] + [""]
rep += [f"[공개 직전 검사 ({'--release: 남아 있으면 실패' if cli.release else '참고용, --release에서만 실패'})]"] \
    + ([f"  {h}" for h in release_hits] or ["  없음"]) + [""]
rep += ["[본문에서 못 찾은 검사 (문장이 바뀌었으면 검사도 고쳐야 함)]"] + (missing or ["  없음"]) + [""]
rep += ["[검증 안 된 숫자 (어떤 검사에도 안 걸림, 사람이 확인)]"] + (leftover or ["  없음"]) + [""]
rep += ["[결과 파일로는 확인 불가, 노트 기록만]"] + (note_only or ["  없음"]) + [""]
rep += ["[참고: 다시 계산한 값]",
        f"  사후 잔차 줄기42 {[round(x, 2) for x in recompute['residuals'][42]]}, 줄기43 {[round(x, 2) for x in recompute['residuals'][43]]}",
        f"  비영어 블록 c1 {nonen[1]}, c3 {nonen[3]} / 960",
        f"  KL 이득 줄기별 {[(round(a, 2), round(b, 2)) for a, b in gain.values()]}, 영역별 비 {[round(v, 2) for v in gain_dom.values()]}",
        f"  grad_norm 평균 {[(k, round(v, 2)) for k, v in gn.items()]}",
        f"  복습 KL 평균 {[(k, round(v, 4)) for k, v in rkl.items()]}, latest_kl 1스텝 로그 {[round(x, 4) for x in kl_floor_latest]}",
        f"  KL 바닥값 (kl_floor.json) latest {KLF['latest_mean']:.5f}, base {KLF['base_mean']:.5f}",
        f"  원본 생성 복습 코드 {code_n}/{gen_n} = {code_n / gen_n:.1%}, 망가진 블록 {sum(broken)}/{len(broken)} = {sum(broken) / len(broken):.1%}",
        f"  중복 블록 {dict((f'{k[0]}/{k[1]}', v) for k, v in dup.items() if v)}",
        f"  옛 한국어 held-out 13-gram 겹침 {old_ko_overlap:.2f}%",
        f"  생성 배치 B {gen_batch['B']}, C {gen_batch['C']}, 스텝 FT {steps_ft}, 복습 {steps_rep}",
        f"  greedy 결과 파일: {SCRATCH_EQ[0] if SCRATCH_EQ else '없음 (검사 생략)'}"]
out = "\n".join(rep)
print(out)
os.makedirs("results", exist_ok=True)
report_path = "results/verify_draft_report" + ("_tex" if IS_TEX else "") + ("_release" if cli.release else "") + ".txt"
open(report_path, "w", encoding="utf-8").write(out + "\n")
print(f"\n저장: {report_path}")
sys.exit(1 if bad or vbad or missing else 0)
