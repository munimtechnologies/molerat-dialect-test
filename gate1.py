# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Step-1 job gate: compute nothing until STEP1-PREREG.md and the step-1 code on GitHub main equal
the job's own copies (CRLF normalised); then fetch the public step-0 helper describe_molerat.py
from main (step 1 reuses its zip reading and trace matching). Waits at most 45 minutes."""
import hashlib
import os
import sys
import time
import urllib.request

RAW = "https://raw.githubusercontent.com/munimtechnologies/molerat-dialect-test/main/"
FILES = ["STEP1-PREREG.md", "molerat_step1.py", "step1_stats.py", "test_step1_stats.py"]
HELPER = "describe_molerat.py"


def norm(blob):
    return blob.replace(b"\r\n", b"\n")


def fetch(name):
    with urllib.request.urlopen(RAW + name + "?t=%d" % int(time.time()), timeout=60) as r:
        return r.read()


def check():
    ok = True
    lines = []
    for name in FILES:
        with open(name, "rb") as fh:
            lm = hashlib.md5(norm(fh.read())).hexdigest()
        try:
            rm = hashlib.md5(norm(fetch(name))).hexdigest()
        except Exception as exc:  # noqa: BLE001
            lines.append("%s local %s remote fetch failed: %s" % (name, lm, exc))
            ok = False
            continue
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
            break
        if time.time() > deadline:
            print("gate failed: public copies missing or different after 45 min; nothing computed",
                  flush=True)
            return 1
        time.sleep(300)
    blob = fetch(HELPER)
    try:
        with open(HELPER, "wb") as fh:
            fh.write(blob)
    except OSError:  # read-only work folder: molerat_step1.py also looks in /tmp/mr
        os.makedirs("/tmp/mr", exist_ok=True)
        with open(os.path.join("/tmp/mr", HELPER), "wb") as fh:
            fh.write(blob)
    print("plan and code are public and identical; %s fetched from main (md5 %s)" % (
        HELPER, hashlib.md5(norm(blob)).hexdigest()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
