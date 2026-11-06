"""
Instances: the Curtois & Qu shift scheduling benchmark (downloaded on first use from a pinned GitHub mirror and
checked by SHA-256), a fictional hospital ward, and rosters read from Excel.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from .problem import read_nrp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache", "nrp")
# the 24 files are identical in two independent mirrors (CLP-NurseRostering and wssuite/NurseScheduler)
MIRROR = "https://raw.githubusercontent.com/GilTeixeira/CLP-NurseRostering/88c5cfa491abcec77edd678beddaede74292f3f0/Dataset/"
SHA = json.load(open(os.path.join(os.path.dirname(__file__), "nrp_sha256.json")))
# best known penalties and lower bounds, schedulingbenchmarks.org (all proven optimal except Instance24)
BEST = {f"Instance{k}": v for k, v in enumerate(
    [607, 828, 1001, 1716, 1143, 1950, 1056, 1300, 439, 4631, 3443, 4040, 1348, 1278, 3828, 3225, 5746, 4459, 3149,
     4769, 21133, 30241, 17285, 40439], 1)}
LOWER = dict(BEST, Instance24=40438)
SIZES = {}


def _fetch(fname: str) -> str:
    path = os.path.join(CACHE, fname)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        os.makedirs(CACHE, exist_ok=True)
        with urllib.request.urlopen(MIRROR + fname, timeout=60) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != SHA[fname]:
        os.remove(path)
        raise ValueError(f"{fname}: the downloaded file does not match its checksum")
    return raw.decode()


def benchmark(name: str):
    inst = read_nrp(_fetch(f"{name}.txt"), name)
    inst.description = (f"{name} (Curtois & Qu benchmark): {inst.E} staff, {inst.S} shift types, {inst.horizon // 7} "
                        f"weeks. Best known penalty {BEST[name]:,}.")
    return inst
