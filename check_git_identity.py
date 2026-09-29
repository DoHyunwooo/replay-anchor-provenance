"""
forgetting-lab / check_git_identity.py

공개 전 검사: 이 저장소에서 커밋에 쓰일 git user.email이 GitHub noreply 주소인지, 이미 있는 커밋의 작성자·커미터 주소도
모두 noreply인지 확인한다. 개인 메일 주소가 공개 커밋 기록에 남는 것을 막기 위한 것. 아무것도 바꾸지 않는다.

  * noreply 형식: <ID>+<username>@users.noreply.github.com (또는 예전 형식 <username>@users.noreply.github.com)
    GitHub > Settings > Emails > "Keep my email addresses private"에 표시된 주소.
  * 주소는 통과·실패와 관계없이 앞부분을 가리고 도메인만 보여 준다 (출력이 어딘가에 붙여 넣어져도 주소가 새지 않게).
  * 결과: 모두 통과하면 exit 0, 하나라도 실패하면 exit 1, git 저장소가 아니면 exit 2.

고치는 법 (이 저장소에만 적용):
  git config user.email "<ID>+<username>@users.noreply.github.com"
이미 개인 주소로 커밋했다면 공개(push) 전에 그 커밋들의 작성자를 고쳐야 한다 (예: git rebase -r --root --exec
"git commit --amend --no-edit --reset-author"; 되돌리기 어려운 작업이므로 백업 후).

예)
  python check_git_identity.py
"""
import re
import subprocess
import sys

NOREPLY = re.compile(r"^(\d+\+)?[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?@users\.noreply\.github\.com$", re.I)


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout.strip()


def mask(addr):
    if "@" not in addr:
        return "(주소 형식 아님)"
    local, domain = addr.rsplit("@", 1)
    return f"{local[:1]}***@{domain}"


def main():
    code, inside = git("rev-parse", "--is-inside-work-tree")
    if code != 0 or inside != "true":
        print("git 저장소가 아님: git init 뒤에 다시 실행하세요.")
        return 2
    fails = []

    # 1. 다음 커밋에 쓰일 주소 (local > global > system 순으로 적용된 값)와 그 출처
    code, email = git("config", "user.email")
    _, origin = git("config", "--show-origin", "user.email")
    where = origin.split("\t")[0] if origin else "(설정 없음)"
    if code != 0 or not email:
        fails.append("user.email이 설정되지 않음")
    elif NOREPLY.match(email):
        print(f"O user.email = {mask(email)} (noreply, 출처 {where})")
    else:
        fails.append(f"user.email이 noreply가 아님: {mask(email)} (출처 {where})")

    # 2. 이미 있는 커밋 (모든 브랜치·태그)의 작성자·커미터 주소
    code, log = git("log", "--all", "--format=%H %ae %ce")
    if code != 0 or not log:
        print("O 커밋 없음 (검사할 기록 없음)")
    else:
        bad = {}
        for line in log.splitlines():
            sha, ae, ce = line.split(" ", 2)
            for who, addr in (("작성자", ae), ("커미터", ce)):
                if not NOREPLY.match(addr):
                    bad.setdefault((who, mask(addr)), []).append(sha[:8])
        n = len(log.splitlines())
        if bad:
            for (who, m), shas in bad.items():
                fails.append(f"커밋 {len(shas)}개의 {who} 주소가 noreply가 아님: {m} (예: {', '.join(shas[:3])})")
        else:
            print(f"O 커밋 {n}개 모두 작성자·커미터 주소가 noreply")

    for f in fails:
        print("X " + f)
    print("통과" if not fails else f"실패 {len(fails)}개: 공개(push) 전에 고치세요 (고치는 법은 이 파일 맨 위 설명).")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
