"""
forgetting-lab / build_lang_data.py

다회차(cycle) 연구(LAB_NOTES 9장) 전용 언어 데이터: 한국어, 일본어, 러시아어, 그리스어 위키백과.
네 언어 모두 같은 절차. 결과는 data/cycles/<lang>_train, data/cycles/<lang>_heldout (+ source.json).
run01~04용 옛 한국어(data/ko_train, data/ko_heldout)는 읽지도 건드리지도 않는다.

  * 학습용 (3840블록): 스트림 처음부터 순서대로, 글마다 토큰 + eos를 이어 붙여 512토큰씩 자른다 (train.py와 같은 방식).
    단 **글 하나는 앞에서부터 최대 20블록 분량(20 x 512토큰)만** 쓰고 끝에 eos를 붙인다. 짧은 글도 그대로 들어간다.
    이어 붙이기라 잘린 글 하나가 블록 21개에 걸칠 수 있다. 글 종류(연도·날짜 글 등)로 빼지 않는다 (실제 위키 분포 유지).
  * held-out (400블록): 학습용에 쓴 마지막 글(일부만 쓰였을 수 있음)의 **다음 글부터**, 글마다 앞에서부터 최대 4블록이 후보.
    512토큰 미만인 글은 건너뛴다. 글 단위로 학습용과 완전히 분리된다.
  * **오염 제거 규칙 (LAB_NOTES 9장, 결과 보기 전 확정 2026-09-26)**: 후보 블록 하나의 13-gram 중 **10%를 넘게** 학습 데이터에
    있으면 그 블록을 빼고, 스트림의 다음 후보로 채운다. 글 종류로는 빼지 않는다. 13-gram은 블록 안에서만, 정확 일치.
  * 검증: 학습용과 held-out의 글 id·제목·같은 블록 겹침이 0이 아니면 저장하지 않는다.
  * 진단 (통과/실패 기준 아님): 언어 전체의 13-gram 겹침 비율, 뺀 블록 수 (후보의 10%를 넘으면 경고),
    학습용·held-out의 글자 수와 공백 기준 단어 수 (같은 512토큰에 담긴 글의 양이 언어마다 달라서).

이미 있으면 덮어쓰지 않는다.

예)
  python build_lang_data.py                 # ko ja ru el 중 없는 것만
  python build_lang_data.py --langs el
"""
import argparse
import datetime
import json
import os

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from replay_quality import NGRAM, gram_set, ngrams

MODEL_NAME = "EleutherAI/pythia-410m"
BLOCK_SIZE = 512
N_TRAIN, N_HELDOUT = 3840, 400
TRAIN_PER_DOC, HELDOUT_PER_DOC = 20, 4
MAX_BLOCK_OVERLAP = 0.10     # held-out 블록의 13-gram 중 학습 데이터에 있는 비율이 이걸 넘으면 뺀다
WARN_DROP_RATE = 0.10        # 후보 중 뺀 비율이 이걸 넘으면 따로 알림
WIKI_CONFIG = "20231101"
OUT_ROOT = "data/cycles"
LANGS = {"ko": "한국어", "ja": "일본어", "ru": "러시아어", "el": "그리스어"}


def text_stats(tok, blocks):
    texts = [tok.decode(b) for b in blocks]
    chars = sum(len(t) for t in texts)
    nonspace = sum(sum(not c.isspace() for c in t) for t in texts)
    words = sum(len(t.split()) for t in texts)
    return {"chars": chars, "nonspace_chars": nonspace, "whitespace_words": words,
            "chars_per_token": round(chars / (len(blocks) * BLOCK_SIZE), 3)}


def save(path, blocks, source):
    Dataset.from_dict({"input_ids": blocks}).save_to_disk(path)
    source.update({"n_blocks": len(blocks), "block_size": BLOCK_SIZE, "tokenizer": MODEL_NAME,
                   "created_at": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(path, "source.json"), "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2, ensure_ascii=False)
    print(f"  -> {path} 저장 ({len(blocks)}블록)")


def build(lang, tok):
    train_dir, held_dir = f"{OUT_ROOT}/{lang}_train", f"{OUT_ROOT}/{lang}_heldout"
    if os.path.isdir(train_dir) or os.path.isdir(held_dir):
        print(f"[{lang}] 이미 있음 ({train_dir} 또는 {held_dir}), 덮어쓰지 않고 건너뜀")
        return None
    config = f"{WIKI_CONFIG}.{lang}"
    print(f"[{lang}] wikimedia/wikipedia {config} 스트리밍")
    stream = iter(load_dataset("wikimedia/wikipedia", config, split="train", streaming=True))

    # 1) 학습용: 글마다 최대 20블록 분량만 잘라서 이어 붙이기
    buffer, train, train_docs, n_capped = [], [], [], 0
    cap = TRAIN_PER_DOC * BLOCK_SIZE
    for a in stream:
        ids = tok(a["text"], verbose=False)["input_ids"] + [tok.eos_token_id]
        if len(ids) > cap:
            ids = ids[:cap - 1] + [tok.eos_token_id]   # 잘라도 글 경계 eos는 유지
            n_capped += 1
        train_docs.append((a["id"], a["title"]))
        buffer.extend(ids)
        while len(buffer) >= BLOCK_SIZE and len(train) < N_TRAIN:
            train.append(buffer[:BLOCK_SIZE])
            buffer = buffer[BLOCK_SIZE:]
        if len(train) >= N_TRAIN:
            break
    if len(train) < N_TRAIN:
        raise SystemExit(f"[{lang}] 학습 블록이 {len(train)}개뿐")
    print(f"  학습용 {len(train)}블록, 글 {len(train_docs)}개, 20블록 분량에서 자른 글 {n_capped}개")
    grams = gram_set(train)

    # 2) held-out: 다음 글부터 글당 앞 최대 4블록이 후보, 13-gram 10% 초과 블록은 빼고 다음 후보로
    held, held_docs, counts, skipped_short = [], [], [], 0
    n_cand, dropped, dropped_docs = 0, [], set()
    for a in stream:
        ids = tok(a["text"], verbose=False)["input_ids"] + [tok.eos_token_id]
        n = min(HELDOUT_PER_DOC, len(ids) // BLOCK_SIZE)
        if n == 0:
            skipped_short += 1
            continue
        kept = 0
        for i in range(n):
            if len(held) >= N_HELDOUT:
                break
            b = ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            g = ngrams(b)
            rate = sum(x in grams for x in g) / len(g)
            n_cand += 1
            if rate > MAX_BLOCK_OVERLAP:
                dropped.append({"title": a["title"], "block_in_doc": i, "overlap_rate": round(rate, 3)})
                dropped_docs.add(a["id"])
                continue
            held.append(b)
            kept += 1
        if kept:
            held_docs.append((a["id"], a["title"]))
            counts.append(kept)
        if len(held) >= N_HELDOUT:
            break
    if len(held) < N_HELDOUT:
        raise SystemExit(f"[{lang}] held-out 블록이 {len(held)}개뿐")
    drop_rate = len(dropped) / n_cand
    print(f"  held-out {len(held)}블록, 글 {len(held_docs)}개 (글당 {min(counts)}~{max(counts)}), "
          f"512토큰 미만이라 건너뛴 글 {skipped_short}개")
    print(f"  오염 제거: 후보 {n_cand}블록 중 {len(dropped)}블록 뺌 ({drop_rate:.1%})"
          + ("  <-- 후보의 10% 초과" if drop_rate > WARN_DROP_RATE else ""))

    # 3) 검증 (글 id·제목·같은 블록 0)
    t_ids, t_titles = {d[0] for d in train_docs}, {d[1] for d in train_docs}
    t_blocks = {tuple(b) for b in train}
    overlap = {"article_id": sum(d[0] in t_ids for d in held_docs),
               "title": sum(d[1] in t_titles for d in held_docs),
               "identical_block": sum(tuple(b) in t_blocks for b in held)}
    print(f"  겹침 검증: id {overlap['article_id']}, 제목 {overlap['title']}, 같은 블록 {overlap['identical_block']}")
    if any(overlap.values()):
        raise SystemExit(f"[{lang}] 학습용과 held-out이 겹칩니다. 저장하지 않고 중단합니다.")

    # 4) 진단: 남은 held-out의 13-gram 겹침
    per_block = [sum(x in grams for x in ngrams(b)) for b in held]
    n_grams = len(held) * (BLOCK_SIZE - NGRAM + 1)
    diag = {"heldout_ngram_overlap_rate": round(sum(per_block) / n_grams, 5),
            "heldout_blocks_with_any_overlap": sum(v > 0 for v in per_block),
            "max_block_overlap_rate": round(max(per_block) / (BLOCK_SIZE - NGRAM + 1), 4)}
    print(f"  진단: held-out 13-gram 중 학습에 있는 비율 {diag['heldout_ngram_overlap_rate']:.2%}, "
          f"겹침 있는 블록 {diag['heldout_blocks_with_any_overlap']}/400, 블록 최대 {diag['max_block_overlap_rate']:.1%}")

    tstats, hstats = text_stats(tok, train), text_stats(tok, held)
    common = {"source": "wikimedia/wikipedia", "config": config, "split": "train", "language": LANGS[lang],
              "study": "LAB_NOTES 9장 다회차 연구 전용"}
    save(train_dir, train, {
        **common, "role": "cycle 학습용",
        "method": f"스트림 처음부터 순서대로, 글마다 토큰+eos를 이어 붙여 512토큰 블록으로 자름. 글 하나는 앞에서부터 "
                  f"최대 {TRAIN_PER_DOC} x 512토큰만 쓰고 끝에 eos. 글 종류로 빼지 않음.",
        "n_docs": len(train_docs), "capped_docs": n_capped, "first_titles": [d[1] for d in train_docs[:10]],
        "first_id": train_docs[0][0], "last_id": train_docs[-1][0], "text_stats": tstats,
    })
    save(held_dir, held, {
        **common, "role": "held-out (학습에 안 씀)",
        "method": f"학습용 마지막 글의 다음 글부터 스트림 순서대로, 글마다 앞에서부터 최대 {HELDOUT_PER_DOC}블록이 후보. "
                  f"512토큰 미만 글은 건너뜀. 후보 블록의 13-gram 중 {MAX_BLOCK_OVERLAP:.0%} 초과가 학습 데이터에 있으면 "
                  f"빼고 다음 후보로 채움 (글 종류로는 빼지 않음).",
        "n_docs": len(held_docs), "blocks_per_doc": {"min": min(counts), "max": max(counts)},
        "skipped_short_docs": skipped_short, "first_titles": [d[1] for d in held_docs[:10]],
        "first_id": held_docs[0][0], "last_id": held_docs[-1][0],
        "overlap_with_train": overlap,
        "contamination_rule": {"max_block_ngram_overlap": MAX_BLOCK_OVERLAP, "candidates": n_cand,
                               "dropped_blocks": len(dropped), "dropped_rate": round(drop_rate, 4),
                               "docs_with_dropped_blocks": len(dropped_docs), "dropped_examples": dropped[:20]},
        "diagnostic_ngram_overlap": diag, "text_stats": hstats,
    })
    return {"lang": lang, "train_docs": len(train_docs), "capped": n_capped, "held_docs": len(held_docs),
            "overlap": overlap, "candidates": n_cand, "dropped": len(dropped), "drop_rate": drop_rate,
            "diag": diag, "train_stats": tstats, "held_stats": hstats}


def main():
    p = argparse.ArgumentParser(description="다회차 연구용 언어 데이터 (ko ja ru el)")
    p.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS))
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.makedirs(OUT_ROOT, exist_ok=True)
    report = [r for r in (build(lang, tok) for lang in args.langs) if r]
    if report:
        path = os.path.join(OUT_ROOT, "build_report.json")
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old + report, f, indent=2, ensure_ascii=False)
        print(f"보고: {path}")


if __name__ == "__main__":
    main()
