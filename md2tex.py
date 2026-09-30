"""
forgetting-lab / md2tex.py

PAPER_DRAFT.md → arXiv용 LaTeX (paper/). 초안을 고치면 이 스크립트를 다시 돌린다.

  * paper/main.tex      : 본문 (article 클래스, natbib 저자-연도 인용, booktabs 표, 그림 4개 PDF)
  * paper/author.tex    : 저자·소속 (이 파일만 고치면 됨. 처음에만 만들고 덮어쓰지 않음)
  * paper/references.bib: BibTeX (확인 상태는 각 항목 위 주석)
  * paper/figures/      : results/figures/*.pdf 복사
  * internal/CLAIM_CHECKLIST.md : 초안의 부록 "Claim checklist" (arXiv 판에서는 뺌, 내부 문서)

규칙: 본문 글자는 바꾸지 않는다 (수치 대조를 LaTeX 본문으로도 하기 위해). 유니코드 기호(−, →, ρ, θ, ‖, ≥, ×)는 그대로 두고
XeLaTeX(Tectonic)에서 newunicodechar로 수식 기호로 찍는다. 인용 "(Author et al., 2026)"은 \\citep/\\citet로 바꾼다.

예)
  python md2tex.py
  tectonic paper/main.tex     # 컴파일 (Tectonic, BibTeX 자동). arXiv는 pdfLaTeX + main.bbl
"""
import os
import re
import shutil

SRC = "PAPER_DRAFT.md"
OUT = "paper"
FIGS = ["fig1_trunks", "fig2_branches", "fig3_age", "fig4_p5"]
SLOPPY_PARAS = ("Code, result files",)   # Overfull \hbox가 나던 문단 (저장소 주소, c_check.json, verify_draft.py)

CITES = [  # (마크다운 표현, LaTeX)
    ("Marek et al. (2026, Eq. 1)", r"\citet[Eq.~1]{marek2026}"),
    ("Marek et al. (2026, Fig. 9)", r"\citet[Fig.~9]{marek2026}"),
    ("Marek et al. (2026)", r"\citet{marek2026}"),
    ("Chen et al. (2026, *Retaining by Doing*, App. A.5)", r"\citet[\emph{Retaining by Doing}, App.~A.5]{chen2026retaining}"),
    ("Chen et al. (2026, *Retaining by Doing*)", r"\citet[\emph{Retaining by Doing}]{chen2026retaining}"),
    ("Chen et al. (2026, *On-Policy Replay*)", r"\citet[\emph{On-Policy Replay}]{chen2026onpolicy}"),   # 다른 저자의 Chen
    ("Wang et al. (2026)", r"\citet{wang2026}"),
    ("(Liu et al., 2026)", r"\citep{liu2026}"),
    ("Huang et al. (2024)", r"\citet{huang2024}"),
    ("(Zhang et al., 2026)", r"\citep{zhang2026}"),
    ("Zhang et al. (2026)", r"\citet{zhang2026}"),
    ("(Resta & Bacciu, 2024)", r"\citep{resta2024}"),
    ("(Masip et al., 2025)", r"\citep{masip2025}"),   # 출판판(PMLR 274, 2025) 우선
    ("(Feng et al., 2026)", r"\citep{feng2026}"),
    ("(Atreya et al., 2026)", r"\citep{atreya2026}"),
    ("Abbes et al. (2026)", r"\citet{abbes2026}"),
    ("Fedus et al. (2020)", r"\citet{fedus2020}"),
    ("Kotha & Liang (2026)", r"\citet{kotha2026}"),
    ("Mahdaviyeh et al. (2026)", r"\citet{mahdaviyeh2026}"),
    ("(Shenfeld et al., 2025)", r"\citep{shenfeld2025}"),
    ("(Shumailov et al., 2024)", r"\citep{shumailov2024}"),
    ("(Biderman et al., 2023)", r"\citep{biderman2023}"),
    ("(Rae et al., 2020)", r"\citep{rae2020}"),
    ("(Raffel et al., 2020)", r"\citep{raffel2020}"),
]

CAPTIONS = {
    "fig1_trunks": "Trunks (B and C1). (a) English perplexity change from the original model (mean of Wikipedia, books and web) after each "
                   "cycle, for FT (no replay), fresh (latest anchor) and base-anchor trunks; solid and dashed lines are the two seeds. "
                   "(b) Korean perplexity change relative to the end of cycle 1.",
    "fig2_branches": "Branches from $M_3$ (C2, C4). Bars are the mean of the two trunks, dots the individual trunks. (a) English change from "
                     "the original model; the dashed line marks the branch start $M_3$. (b) Mean Korean/Japanese/Russian change from $M_3$. "
                     "Dotted bars: English-prose filtered replay; hatched bars: KL objective on the same replay text as the NTP bar.",
    "fig3_age": "Anchor age (C3). (a) English change versus the age of the anchor that generated the replay (0 = $M_3$, 3 = original model); "
                "dotted lines with open markers show the post hoc description ``base\\_full plus the anchor's own English drift''. "
                "(b) Intermediate-language change from $M_3$. The 50/50 mix has no single anchor age; it is shown in a separate narrow "
                "panel to the right of each age panel, off the age axis.",
    "fig4_p5": "English forgetting against (a) probe KL(original $\\Vert$ checkpoint) and (b) relative parameter distance, over trunk "
               "checkpoints and branches. Spearman $\\rho$ is computed on the pre-specified set (trunks and C2 branches); C3/C4 branches "
               "(open markers) are shown for reference. Points share trunks and are not independent.",
}

BIB = r"""% 서지는 2026-09-29에 원문 페이지(arXiv, ACL Anthology, PMLR, OpenReview)에서 저자가 확인한 값. 미확인 필드가 생기면 항목 위 주석에 "[확인 필요]"로 적는다 (verify_draft.py --release가 잡음).
% 제목은 모두 {{...}}로 감싸 원래 대소문자를 유지한다 (plainnat이 소문자로 바꾸지 않게).
% 제목 대소문자는 2026-09-30에 원문(출판판 우선: PMLR·Crossref, 없으면 arXiv API)과 글자 그대로 대조함.
% 출판판과 arXiv판이 둘 다 있으면 출판 정보를 쓰고 arXiv id는 주석으로만 남긴다 (plainnat은 eprint를 찍지 않음).

% [confirmed] 저자, 제목, arXiv id
@misc{marek2026,
  title         = {{Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay}},
  author        = {Marek, Martin and Cho, Dongkyu and Qiu, Shikai and Chunara, Rumi and Izmailov, Pavel and Wilson, Andrew Gordon},
  year          = {2026},
  eprint        = {2605.26097},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2605.26097}
}

% [confirmed] 원문 확인 (2026-09-28): 저자 전체, 제목, arXiv id. Retaining by Doing의 Chen과 다른 저자 → 키 구분
@misc{chen2026onpolicy,
  title         = {{On-Policy Replay for Continual Supervised Fine-Tuning}},
  author        = {Chen, Yan and Zhu, Taojie and Zhang, Meng and Chen, Xin and Huang, Jiaqi and Xu, Dongyang and Wang, Yizhi},
  year          = {2026},
  eprint        = {2605.29495},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2605.29495}
}

% [confirmed] 원문 확인 (2026-09-28): 저자 전체, 제목, arXiv id
@misc{wang2026,
  title         = {{Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training}},
  author        = {Wang, Meng and Zhao, Haohan and Liu, Wenzhuo and Yang, Lu and Liu, Geng and Guo, Haiyang and Xie, Guo-Sen and Meng, Gaofeng and Liu, Hongbin and Zhu, Fei},
  year          = {2026},
  eprint        = {2607.01763},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2607.01763}
}

% [confirmed] 원문 확인 (2026-09-28): 저자 전체, 제목, arXiv id
@misc{liu2026,
  title         = {{Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting}},
  author        = {Liu, Chi and Chen, Xin and Zhou, Xu and Tu, Fangbo and Manoharan, Srinivasan},
  year          = {2026},
  eprint        = {2604.15794},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2604.15794}
}

% [confirmed] 원문 확인 (2026-09-29): 저자 전체, 제목, arXiv id (ComposeCL)
@misc{zhang2026,
  title         = {{Continual Learning Mechanisms Compose for Long-Horizon Memorization}},
  author        = {Zhang, Zheyuan and Zhang, Alvin and Khashabi, Daniel and Shu, Tianmin},
  year          = {2026},
  eprint        = {2609.06986},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2609.06986}
}

% [confirmed] 저자 전체, 제목, 학회, 쪽 (PMLR 330)
@inproceedings{abbes2026,
  title     = {{Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models}},
  author    = {Abbes, Istabrak and Subbaraj, Gopeshh and Riemer, Matthew and Islah, Nizar and Tabaru, Tsuguchika and Kingetsu, Hiroaki and Chandar, Sarath and Rish, Irina},
  booktitle = {Proceedings of the 4th Conference on Lifelong Learning Agents},
  series    = {Proceedings of Machine Learning Research},
  volume    = {330},
  pages     = {465--486},
  year      = {2026}
}

% [confirmed] 저자, 제목, 학회 (ICML 2026), arXiv id. PMLR 쪽수 없이 인용 (저자 결정, 2026-09-29)
@inproceedings{chen2026retaining,
  title     = {{Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting}},
  author    = {Chen, Howard and Razin, Noam and Narasimhan, Karthik and Chen, Danqi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2510.18874}
}

% [confirmed] 저자, 제목, 학회, 쪽, DOI. arXiv:2403.01244 (출판판 우선이라 주석으로만)
@inproceedings{huang2024,
  title     = {{Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal}},
  author    = {Huang, Jianheng and Cui, Leyang and Wang, Ante and Yang, Chengyi and Liao, Xinting and Song, Linfeng and Yao, Junfeng and Su, Jinsong},
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers},
  pages     = {1416--1428},
  year      = {2024},
  doi       = {10.18653/v1/2024.acl-long.77}
}

% [confirmed] 저자, 제목, 학회, 쪽 (PMLR 274). 출판판 우선 → 2025 (arXiv:2311.14028은 주석으로만)
@inproceedings{masip2025,
  title     = {{Continual Learning of Diffusion Models with Generative Distillation}},
  author    = {Masip, Sergi and Rodriguez, Pau and Tuytelaars, Tinne and van de Ven, Gido M.},
  booktitle = {Proceedings of The 3rd Conference on Lifelong Learning Agents},
  series    = {Proceedings of Machine Learning Research},
  volume    = {274},
  pages     = {431--456},
  year      = {2025}
}

% [confirmed] 저자, 제목, arXiv id
@misc{shenfeld2025,
  title         = {{RL's Razor: Why Online Reinforcement Learning Forgets Less}},
  author        = {Shenfeld, Idan and Pari, Jyothish and Agrawal, Pulkit},
  year          = {2025},
  eprint        = {2509.04259},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2509.04259}
}

% [confirmed] 저자, 제목, arXiv id. 학회 게재는 확인되지 않아 arXiv preprint로 인용 (저자 결정, 2026-09-29)
@misc{kotha2026,
  title         = {{Replaying pre-training data improves fine-tuning}},
  author        = {Kotha, Suhas and Liang, Percy},
  year          = {2026},
  eprint        = {2603.04964},
  archivePrefix = {arXiv},
  howpublished  = {arXiv preprint arXiv:2603.04964}
}

% [confirmed] 저자 전체, 제목, 학회, 쪽 (PMLR 330, pp. 15-42, 2026)
@inproceedings{mahdaviyeh2026,
  title     = {{Replay can provably increase forgetting}},
  author    = {Mahdaviyeh, Yasaman and Lucas, James and Ren, Mengye and Tolias, Andreas S. and Zemel, Richard and Pitassi, Toniann},
  booktitle = {Proceedings of the 4th Conference on Lifelong Learning Agents},
  series    = {Proceedings of Machine Learning Research},
  volume    = {330},
  pages     = {15--42},
  year      = {2026}
}

% [confirmed] 저자, 제목, 학회 (NAACL 2024), 쪽, DOI. arXiv:2403.13130 (출판판 우선이라 주석으로만).
@inproceedings{resta2024,
  title     = {{Self-generated Replay Memories for Continual Neural Machine Translation}},
  author    = {Resta, Michele and Bacciu, Davide},
  booktitle = {North American Chapter of the Association for Computational Linguistics (NAACL)},
  pages     = {175--191},
  year      = {2024},
  doi       = {10.18653/v1/2024.naacl-long.10}
}

% [confirmed] 저자 전체, 제목, 학회, 쪽, DOI
@inproceedings{feng2026,
  title     = {{FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning}},
  author    = {Feng, Yujie and Wang, Hao and Li, Jian and Chu, Xu and Kang, Zhaolu and Liu, Yiran and Wang, Yasha and Yu, Philip S. and Wu, Xiao-Ming},
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers},
  pages     = {24945--24965},
  year      = {2026},
  doi       = {10.18653/v1/2026.acl-long.1144}
}

% [confirmed] 저자 전체, 제목, arXiv id
@misc{atreya2026,
  title         = {{When to Review: Spaced Repetition for Continual Pre-Training of Language Models}},
  author        = {Atreya, Alankar and Batra, Devesh and Mantri, Yoages Kumar and Bantug, Geremy and Cowan, Greig A. and Khraishi, Raad},
  year          = {2026},
  eprint        = {2608.17530},
  archivePrefix = {arXiv},
  howpublished  = {arXiv:2608.17530}
}

% [confirmed] 저자 전체, 제목, 학회, 쪽 (PMLR 119)
@inproceedings{fedus2020,
  title     = {{Revisiting Fundamentals of Experience Replay}},
  author    = {Fedus, William and Ramachandran, Prajit and Agarwal, Rishabh and Bengio, Yoshua and Larochelle, Hugo and Rowland, Mark and Dabney, Will},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {3061--3071},
  year      = {2020}
}

% [confirmed] 저자 전체, 제목, 학술지, 권·호·쪽, DOI
@article{shumailov2024,
  title   = {{AI models collapse when trained on recursively generated data}},
  author  = {Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Papernot, Nicolas and Anderson, Ross and Gal, Yarin},
  journal = {Nature},
  volume  = {631},
  number  = {8022},
  pages   = {755--759},
  year    = {2024},
  doi     = {10.1038/s41586-024-07566-y}
}

% [confirmed] 저자 전체, 제목, 학회, 쪽 (PMLR 202)
@inproceedings{biderman2023,
  title     = {{Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling}},
  author    = {Biderman, Stella and Schoelkopf, Hailey and Anthony, Quentin Gregory and Bradley, Herbie and O'Brien, Kyle and Hallahan, Eric and Khan, Mohammad Aflah and Purohit, Shivanshu and Prashanth, Usvsn Sai and Raff, Edward and Skowron, Aviya and Sutawika, Lintang and Van Der Wal, Oskar},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {2397--2430},
  year      = {2023}
}

% [confirmed] 저자 5명 (ICLR 2020 OpenReview 판 기준; arXiv:1911.05507 목록에는 Hillier가 빠져 있어 arXiv BibTeX는 쓰지 않음). (PG-19)
@inproceedings{rae2020,
  title     = {{Compressive Transformers for Long-Range Sequence Modelling}},
  author    = {Rae, Jack W. and Potapenko, Anna and Jayakumar, Siddhant M. and Hillier, Chloe and Lillicrap, Timothy P.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

% [confirmed] 저자 전체, 제목, 학술지, 권·호·쪽, 연도. (C4)
@article{raffel2020,
  title   = {{Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer}},
  author  = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal = {Journal of Machine Learning Research},
  volume  = {21},
  number  = {140},
  pages   = {1--67},
  year    = {2020}
}
"""

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
% arXiv는 pdfLaTeX로 컴파일한다 (XeLaTeX 비공식). 로컬 Tectonic은 XeLaTeX. 엔진에 따라 나눈다.
\usepackage{iftex}
\ifPDFTeX
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
  \usepackage{textcomp}
\else
  \usepackage{fontspec}
\fi
\usepackage{newunicodechar}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,tabularx}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small}
% 본문의 유니코드 기호를 그대로 둔다 (수치 대조용). 수식 기호로 찍는다.
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{‖}{\ensuremath{\Vert}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newcommand{\paramdist}{\ensuremath{\lVert\theta-\theta_0\rVert/\lVert\theta_0\rVert}}
\input{author}
"""

AUTHOR = r"""% 저자·소속은 이 파일만 고치면 된다 (md2tex.py는 이 파일이 있으면 덮어쓰지 않음)
\newcommand{\paperauthor}{Hyunwoo Do}
\newcommand{\paperaffil}{Independent Researcher}
\newcommand{\paperemail}{}
"""


def esc(s):
    """본문 텍스트의 LaTeX 특수문자 (인용·수식 처리 전)."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def inline(s):
    # 인라인 코드 먼저 (안쪽은 그대로 이스케이프)
    codes = []

    def keep_code(m):
        codes.append(r"\texttt{" + m.group(1).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&") + "}")
        return f"\x00{len(codes) - 1}\x00"
    s = re.sub(r"`([^`]+)`", keep_code, s)
    # 주소는 \url{} (끝의 마침표·쉼표는 문장 부호로 남긴다)
    s = re.sub(r"https?://[^\s)]*[^\s).,;:]", lambda m: (codes.append(r"\url{" + m.group(0) + "}"), f"\x00{len(codes) - 1}\x00")[1], s)
    s = s.replace("‖θ − θ0‖ / ‖θ0‖", "\x01PD\x01")
    cites = []
    for md, tex in CITES:
        if md in s:
            cites.append(tex)
            s = s.replace(md, f"\x02{len(cites) - 1}\x02")
    s = esc(s)
    s = re.sub(r"\bM_([0-9j])\b", r"$M_\1$", s)
    s = s.replace("_", r"\_").replace(r"$M\_", "$M_")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r'"([^"]+)"', r"``\1''", s)
    s = s.replace("\x01PD\x01", r"\paramdist{}")
    s = re.sub(r"\x02(\d+)\x02", lambda m: cites[int(m.group(1))], s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)
    return s


def table(rows, caption, label):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s\-|]+\|$", r.strip())]
    n = len(cells[0])
    wide = {"tab:1": "l" + "X" * (n - 1), "tab:3": "l" * (n - 1) + "X"}    # 글이 긴 표는 폭을 맞춰 줄바꿈
    spec = wide.get(label, "l" * n)
    env = "tabularx}{\\linewidth" if label in wide else "tabular"
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + inline(caption) + "}", r"\label{" + label + "}",
           r"\begin{" + env + "}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in cells[0]) + r" \\")
    out.append(r"\midrule")
    for r in cells[1:]:
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{" + env.split("}")[0] + "}", r"\end{table}"]
    return out


def figure(name, n):
    return [r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figures/" + name + ".pdf}",
            r"\caption{" + CAPTIONS[name] + "}", r"\label{fig:" + str(n) + "}", r"\end{figure}", ""]


def main():
    md = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    body, rest = md.split("\n## References", 1)
    # 참고문헌 뒤: 부록 A (arXiv 판에 넣음, \appendix 뒤) + 체크리스트 (내부 문서로 뺌)
    refs_and_app, checklist = rest.split("\n## Appendix: Claim checklist", 1)
    if "\n## Appendix A. " in refs_and_app:
        body += "\n## \x04" + refs_and_app.split("\n## Appendix A. ", 1)[1]
    os.makedirs("internal", exist_ok=True)
    with open("internal/CLAIM_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write("# Claim checklist (internal, not in the arXiv version)\n\n"
                "Extracted from PAPER_DRAFT.md by md2tex.py.\n\n## Appendix: Claim checklist" + checklist)

    lines = body.split("\n")
    title = lines[0].lstrip("# ").strip()
    out, i = [], 1
    in_list = None
    tags_para = None
    pending_caption = None
    abstract = False
    fig_after = {"### 5.1": ["fig1_trunks", "fig2_branches"], "**Anchor age (C3)": ["fig3_age"], "### 5.4": ["fig4_p5"]}
    fig_no = {n: k + 1 for k, n in enumerate(FIGS)}
    placed = set()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{" + in_list + "}")
            in_list = None

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("*Workshop draft"):
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if s == "---":
            i += 1
            continue
        if s.startswith("## "):
            close_list()
            if abstract:
                out.append(r"\end{abstract}")
                abstract = False
                if tags_para:
                    out += [tags_para, ""]
                    tags_para = None
            head = s[3:]
            if head == "Abstract":
                out.append(r"\begin{abstract}")
                abstract = True
            elif head.startswith("\x04"):                       # 부록 A: 참고문헌 뒤로 보낸다
                out.append("%%APPENDIX%%")
                out.append(r"\section{" + inline(head[1:].strip()) + "}")
            elif not head[0].isdigit():                          # Code and data availability, Acknowledgments
                out.append(r"\section*{" + inline(head) + "}")
            else:
                out.append(r"\section{" + inline(re.sub(r"^\d+\.\s*", "", head)) + "}")
            i += 1
            continue
        if s.startswith("### "):
            close_list()
            out.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", s[4:])) + "}")
            for key, names in fig_after.items():
                if s.startswith(key):
                    for nme in names:
                        out += figure(nme, fig_no[nme]); placed.add(nme)
            i += 1
            continue
        m = re.match(r"^\*\*(Table \d)\. (.+?)\*\*(.*)$", s)
        if m:
            cap = m.group(2) + m.group(3)
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith("|"):
                i += 1
                cap += " " + lines[i].strip()
            pending_caption = (cap.strip(), "tab:" + m.group(1)[-1])
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            cap, lab = pending_caption or ("", "tab:x")
            out += table(rows, cap, lab)
            pending_caption = None
            continue
        m_num = re.match(r"^(\d+)\.\s+(.*)$", s)
        m_bul = re.match(r"^-\s+(.*)$", s)
        if m_num or m_bul:
            kind = "enumerate" if m_num else "itemize"
            if in_list != kind:
                close_list(); out.append(r"\begin{" + kind + "}"); in_list = kind
            item = (m_num or m_bul).group(2 if m_num else 1)
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip():
                i += 1
                item += " " + lines[i].strip()
            out.append(r"\item " + inline(item))
            i += 1
            continue
        if not s:
            close_list()
            out.append("")
            i += 1
            continue
        # 보통 문단 (연속된 줄을 한 문단으로)
        para = s
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||-\s|\d+\.\s|\*\*Table|---)", lines[i + 1].strip()):
            i += 1
            para += " " + lines[i].strip()
        if para.startswith("*Tags used throughout"):
            # 초록 뒤에 둔다 (마크다운에서는 초록 앞에 있음)
            tags_para = r"\noindent{\small " + inline(para.strip("*")) + "}"
        elif para.startswith(SLOPPY_PARAS):
            # 긴 주소·파일명이 줄 끝을 넘는 문단: 이 문단에만 \sloppy (다른 쪽 줄바꿈은 그대로)
            out.append(r"{\sloppy " + inline(para) + r"\par}")
        else:
            out.append(inline(para))
        for key, names in fig_after.items():
            if para.startswith(key):
                for nme in names:
                    if nme not in placed:
                        out += [""] + figure(nme, fig_no[nme]); placed.add(nme)
        i += 1
    close_list()
    if abstract:
        out.append(r"\end{abstract}")

    main_part, app_part = (out[:out.index("%%APPENDIX%%")], out[out.index("%%APPENDIX%%") + 1:]) \
        if "%%APPENDIX%%" in out else (out, [])
    doc = [PREAMBLE, r"\title{" + inline(title) + "}",
           r"\author{\paperauthor\\ \paperaffil\ifx\paperemail\empty\else\\ \texttt{\paperemail}\fi}", r"\date{}",
           r"\begin{document}", r"\maketitle", ""] + main_part + ["", r"\bibliographystyle{plainnat}", r"\bibliography{references}", ""] \
        + ([r"\appendix", ""] + app_part if app_part else []) + [r"\end{document}", ""]
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    with open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    with open(os.path.join(OUT, "references.bib"), "w", encoding="utf-8") as f:
        f.write(BIB)
    if not os.path.exists(os.path.join(OUT, "author.tex")):
        with open(os.path.join(OUT, "author.tex"), "w", encoding="utf-8") as f:
            f.write(AUTHOR)
    for nme in FIGS:
        shutil.copy(os.path.join("results", "figures", nme + ".pdf"), os.path.join(OUT, "figures", nme + ".pdf"))
    missing = [n for n in FIGS if n not in placed]
    print(f"저장: {OUT}/main.tex, references.bib, author.tex, figures/ (그림 {len(placed)}개 배치"
          + (f", 배치 못 함: {missing}" if missing else "") + "), internal/CLAIM_CHECKLIST.md")


if __name__ == "__main__":
    main()
