"""
forgetting-lab / build_en_heldout.py

영어를 "영역별로" 얼마나 잊었는지 재기 위한 held-out 세트 3개를 만든다.
각 400블록 x 512토큰, 한국어 held-out과 같은 토크나이저 / 글 끝 eos / 블록 자르기.

  data/en_heldout_wiki   영어 위키백과. data/en_replay(run03 복습 데이터)에 쓴 글 "다음" 글부터,
                         글마다 앞에서부터 최대 4블록 (100개 이상 글에서 400블록).
                         replay를 똑같이 다시 만들어 원본과 일치하는지 확인하고,
                         id·제목·블록 단위로 겹침이 0인지 검증한다.
  data/en_heldout_books  PG-19 (구텐베르크 책) test split, parquet 복제본 emozilla/pg19.
                         BookCorpus는 LAMBADA와 같은 계열이라 쓰지 않는다.
                         책마다 앞 20% 지점부터 연속 블록을 뽑고, 여러 권에서 고르게
                         (한 바퀴에 책마다 1블록씩, 책당 최대 20블록).
  data/en_heldout_web    allenai/c4 en validation 첫 파일, 문서 순서대로.

각 폴더에 source.json(출처, 글/책 수, 첫 제목들)을 남긴다.
이미 있는 세트는 절대 덮어쓰지 않는다 (덮어쓰면 기존 평가 결과와 비교가 깨짐).

예)
  python build_en_heldout.py                 # 없는 것만 만든다
  python build_en_heldout.py --only books
"""
import argparse
import gzip
import json
import os

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-410m"
BLOCK_SIZE = 512
N_BLOCKS = 400
REPLAY_DIR = "data/en_replay"
REPLAY_BLOCKS = 2400
OUT = {"wiki": "data/en_heldout_wiki", "books": "data/en_heldout_books",
       "web": "data/en_heldout_web"}

BOOKS_REPO = "emozilla/pg19"
BOOKS_FILE = "data/test-00000-of-00001-29a571947c0b5ccc.parquet"
BOOK_START = 0.2           # 책 앞부분(서문·목차)을 피해서 20% 지점부터
MAX_PER_BOOK = 20
WIKI_PER_DOC = 4           # 위키 held-out: 글당 최대 블록 수 (주제 다양성)
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"


def concat_blocks(docs, tokenizer, n_blocks):
    """train.py의 build_wiki_blocks와 같은 방식: 글마다 토큰 + eos를 이어 붙여 자른다.
    docs: (메타, 본문) 반복자. 반환: 블록들, 사용한 글의 메타 목록 (마지막 글은 일부만 쓰였을 수 있음)."""
    buffer, blocks, used = [], [], []
    for meta, text in docs:
        used.append(meta)
        ids = tokenizer(text, verbose=False)["input_ids"]
        buffer.extend(ids + [tokenizer.eos_token_id])
        while len(buffer) >= BLOCK_SIZE:
            blocks.append(buffer[:BLOCK_SIZE])
            buffer = buffer[BLOCK_SIZE:]
        if len(blocks) >= n_blocks:
            break
    if len(blocks) < n_blocks:
        raise SystemExit(f"블록이 {len(blocks)}개밖에 안 나옵니다 ({n_blocks}개 필요)")
    return blocks[:n_blocks], used


def save(name, blocks, source):
    path = OUT[name]
    Dataset.from_dict({"input_ids": blocks}).save_to_disk(path)
    source.update({"n_blocks": len(blocks), "block_size": BLOCK_SIZE,
                   "tokenizer": MODEL_NAME})
    with open(os.path.join(path, "source.json"), "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2, ensure_ascii=False)
    print(f"  -> {path} 저장 ({len(blocks)}블록)")


# ---------- 위키: replay 다음 글부터 ----------

def build_wiki(tokenizer):
    stream = iter(load_dataset("wikimedia/wikipedia", "20231101.en",
                               split="train", streaming=True))
    docs = (((a["id"], a["title"]), a["text"]) for a in stream)

    # 1) run03의 replay를 똑같이 다시 만들어서 어떤 글을 썼는지 알아낸다
    print("  replay 재현 중 (어떤 글을 썼는지 확인)...")
    replay_blocks, replay_used = concat_blocks(docs, tokenizer, REPLAY_BLOCKS)
    saved = load_from_disk(REPLAY_DIR)["input_ids"]
    if len(saved) != REPLAY_BLOCKS or any(list(a) != b for a, b in zip(saved, replay_blocks)):
        raise SystemExit("다시 만든 replay가 data/en_replay와 다릅니다. "
                         "데이터 소스가 바뀌었을 수 있어 겹침 검증이 불가능하니 중단합니다.")
    print(f"  replay 2400블록 완전 일치, 사용한 글 {len(replay_used)}개 "
          f"(마지막: {replay_used[-1][1]!r}, 일부만 사용)")

    # 2) 같은 스트림을 이어서 = replay 마지막 글의 다음 글부터.
    #    주제가 몇 개 긴 글에 몰리지 않게 글마다 앞에서부터 최대 WIKI_PER_DOC 블록만 쓴다
    #    (글 하나 안에서만 자르므로 512토큰이 안 되는 짧은 글은 건너뜀)
    blocks, used, counts, skipped_short = [], [], [], 0
    for meta, text in docs:
        ids = tokenizer(text, verbose=False)["input_ids"] + [tokenizer.eos_token_id]
        n = min(WIKI_PER_DOC, len(ids) // BLOCK_SIZE, N_BLOCKS - len(blocks))
        if n == 0:
            skipped_short += 1
            continue
        blocks += [ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] for i in range(n)]
        used.append(meta)
        counts.append(n)
        if len(blocks) >= N_BLOCKS:
            break
    if len(blocks) < N_BLOCKS:
        raise SystemExit(f"위키 블록이 {len(blocks)}개밖에 안 나옵니다")
    print(f"  글 {len(used)}개에서 {len(blocks)}블록 (글당 {min(counts)}~{max(counts)}), "
          f"512토큰 미만이라 건너뛴 짧은 글 {skipped_short}개")

    # 3) 겹침 검증
    r_ids, r_titles = {m[0] for m in replay_used}, {m[1] for m in replay_used}
    h_ids, h_titles = {m[0] for m in used}, {m[1] for m in used}
    r_blocks = {tuple(b) for b in replay_blocks}
    overlap = {"article_id": len(r_ids & h_ids), "title": len(r_titles & h_titles),
               "identical_block": sum(tuple(b) in r_blocks for b in blocks)}
    print(f"  겹침 검증: id {overlap['article_id']}개, 제목 {overlap['title']}개, "
          f"같은 블록 {overlap['identical_block']}개")
    if any(overlap.values()):
        raise SystemExit("replay와 겹치는 글이 있습니다. 저장하지 않고 중단합니다.")

    save("wiki", blocks, {
        "source": "wikimedia/wikipedia", "config": "20231101.en", "split": "train",
        "method": f"data/en_replay에 쓴 글 바로 다음 글부터 스트림 순서대로, 글마다 "
                  f"토큰+eos를 앞에서부터 512토큰 블록으로 잘라 최대 {WIKI_PER_DOC}블록 "
                  f"(주제 다양성 확보). 512토큰 미만인 글은 건너뜀.",
        "n_docs": len(used),
        "blocks_per_doc": {"min": min(counts), "max": max(counts),
                           "docs_with_max": sum(c == WIKI_PER_DOC for c in counts)},
        "skipped_short_docs": skipped_short,
        "first_titles": [m[1] for m in used[:10]],
        "version": "v2 (2026-09-25). v1은 글을 이어 붙이는 방식이라 글 40개뿐이었음",
        "first_id": used[0][0], "last_id": used[-1][0],
        "replay_check": {"replay_articles": len(replay_used),
                         "replay_last_title": replay_used[-1][1],
                         "replay_reproduced_exactly": True, "overlap": overlap},
    })


# ---------- 책: PG-19 test ----------

def build_books(tokenizer):
    path = hf_hub_download(BOOKS_REPO, BOOKS_FILE, repo_type="dataset")
    df = pd.read_parquet(path)
    print(f"  PG-19 test: 책 {len(df)}권 (컬럼 {list(df.columns)})")

    per_book = []   # 책마다 뽑을 수 있는 연속 블록들 (최대 MAX_PER_BOOK)
    for _, row in df.iterrows():
        text = row["text"]
        start = text.find("\n\n", int(len(text) * BOOK_START))  # 문단 경계에서 시작
        start = int(len(text) * BOOK_START) if start < 0 else start + 2
        # 필요한 만큼만 토큰화 (블록 20개 ≈ 1만 토큰 ≈ 5~6만 글자면 충분)
        chunk = text[start:start + MAX_PER_BOOK * BLOCK_SIZE * 6]
        ids = tokenizer(chunk, verbose=False)["input_ids"] + [tokenizer.eos_token_id]
        n = min(MAX_PER_BOOK, len(ids) // BLOCK_SIZE)
        per_book.append([ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] for i in range(n)])

    # 한 바퀴에 책마다 1블록씩 -> 400개가 여러 권에 고르게 퍼진다
    blocks, taken = [], [0] * len(per_book)
    k = 0
    while len(blocks) < N_BLOCKS:
        progressed = False
        for i, bb in enumerate(per_book):
            if k < len(bb) and len(blocks) < N_BLOCKS:
                blocks.append(bb[k])
                taken[i] += 1
                progressed = True
        if not progressed:
            raise SystemExit(f"책에서 블록이 {len(blocks)}개밖에 안 나옵니다")
        k += 1
    used = [i for i, t in enumerate(taken) if t]
    title_col = "short_book_title" if "short_book_title" in df.columns else df.columns[0]

    save("books", blocks, {
        "source": f"{BOOKS_REPO} (deepmind/pg19의 parquet 복제본)", "split": "test",
        "file": BOOKS_FILE,
        "method": f"책마다 본문 {int(BOOK_START * 100)}% 지점 다음 문단부터 연속 512토큰 블록. "
                  f"책마다 1블록씩 돌아가며 뽑아 400개 (책당 최대 {MAX_PER_BOOK}).",
        "n_docs": len(used), "blocks_per_book": {"min": min(taken[i] for i in used),
                                                 "max": max(taken)},
        "first_titles": [str(df.iloc[i][title_col]) for i in used[:10]],
        "note": "BookCorpus(LAMBADA 계열)는 일부러 제외. PG-19는 1919년 이전 책이라 "
                "LAMBADA의 현대 소설과 문체가 다를 수 있고, Pile(Pythia 사전학습)에 "
                "PG-19가 들어 있어 사전학습 때 본 책일 수 있음.",
    })


# ---------- 웹: C4 en validation ----------

def build_web(tokenizer):
    path = hf_hub_download("allenai/c4", C4_FILE, repo_type="dataset")

    def docs():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield d["url"], d["text"]

    blocks, used = concat_blocks(docs(), tokenizer, N_BLOCKS)
    save("web", blocks, {
        "source": "allenai/c4", "config": "en", "split": "validation", "file": C4_FILE,
        "method": "파일 순서대로 문서마다 토큰+eos를 이어 붙여 512토큰으로 자름 (train.py와 동일)",
        "n_docs": len(used), "first_titles": used[:10],
        "note": "C4 문서에는 제목이 없어서 first_titles 자리에 URL을 적음",
    })


def main():
    p = argparse.ArgumentParser(description="영어 영역별 held-out 세트 만들기")
    p.add_argument("--only", nargs="+", choices=list(OUT), help="이것만 만들기")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    builders = {"wiki": build_wiki, "books": build_books, "web": build_web}
    for name in args.only or list(OUT):
        print(f"[{name}] {OUT[name]}")
        if os.path.isdir(OUT[name]):
            print("  이미 있음, 덮어쓰지 않고 건너뜀")
            continue
        builders[name](tokenizer)


if __name__ == "__main__":
    main()
