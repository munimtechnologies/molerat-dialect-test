# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Job gate: compute nothing until the plan and code on GitHub main equal the job's own copies
(CRLF normalised). Waits at most 45 minutes, then exits 1."""
import hashlib
import sys
import time
import urllib.request

RAW = "https://raw.githubusercontent.com/munimtechnologies/molerat-dialect-test/main/"
FILES = ["DESCRIBE-PLAN.md", "describe_molerat.py", "test_describe_molerat.py"]


def norm(blob):
    return blob.replace(b"\r\n", b"\n")


def check():
    ok = True
    lines = []
    for name in FILES:
        with open(name, "rb") as fh:
            local = norm(fh.read())
        lm = hashlib.md5(local).hexdigest()
        try:
            with urllib.request.urlopen(RAW + name + "?t=%d" % int(time.time()), timeout=60) as r:
                remote = norm(r.read())
        except Exception as exc:  # noqa: BLE001
            lines.append("%s local %s remote fetch failed: %s" % (name, lm, exc))
            ok = False
            continue
        rm = hashlib.md5(remote).hexdigest()
        same = lm == rm
        lines.append("%s local %s remote %s %s" % (name, lm, rm, "same" if same else "DIFFERENT"))
        ok = ok and same
    return ok, lines


def main():
    deadline = time.time() + 45 * 60
    while True:
        ok, lines = check()
        print("\n".join(lines), flush=True)
        if ok:
            print("plan and code are public and identical", flush=True)
            return 0
        if time.time() > deadline:
            print("gate failed: public copies missing or different after 45 min; nothing computed",
                  flush=True)
            return 1
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
